from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from scipy.stats import kurtosis, pearsonr, skew, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET_SPECS = {
    10: ("dataset_target_10", "target_log_return_10m"),
    20: ("dataset_target_20/with_price_hmm_n4", "target_log_return_20m"),
    30: ("dataset_target_30", "target_log_return_30m"),
}
DEFAULT_BUCKET = "binance-data-downloader"
DEFAULT_RESULTS_SUBDIR = "stage1_architecture_search"
DEFAULT_DIRECTION_THRESHOLD = 0.0025
DEFAULT_HMM_FEATURE_PREFIX = "price_hmm_n4"
QUANTILE_ALPHAS = (0.05, 0.25, 0.50, 0.75, 0.95)


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "build_price_feature_day.py").exists():
            return candidate
    raise FileNotFoundError("Could not find project root with build_price_feature_day.py")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TARGET_EXPERIMENT_DIR = PROJECT_ROOT / "analysis" / "target_10_model_experiments"
if str(TARGET_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(TARGET_EXPERIMENT_DIR))

from build_price_feature_day import load_s3_parquet, make_s3_client
from train_target_10_models import available_features, selected_experiment_features


@dataclass(frozen=True)
class PreparedData:
    feature_columns: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    train_timestamps: np.ndarray | None
    test_timestamps: np.ndarray | None


@dataclass(frozen=True)
class SearchJob:
    job_id: str
    model_family: str
    loss_function: str
    params: dict[str, Any]
    alpha: float | None = None


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("stage1_architecture_search")


def object_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True


def upload_bytes(s3, bucket: str, key: str, payload: bytes, content_type: str) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)


def upload_json(s3, bucket: str, key: str, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    upload_bytes(s3, bucket, key, body, "application/json")


def upload_parquet(s3, bucket: str, key: str, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, engine="pyarrow", compression="zstd")
    upload_bytes(s3, bucket, key, buffer.getvalue(), "application/vnd.apache.parquet")


def upload_csv(s3, bucket: str, key: str, frame: pd.DataFrame) -> None:
    upload_bytes(s3, bucket, key, frame.to_csv(index=False).encode("utf-8-sig"), "text/csv")


def serialize_model(model: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    return buffer.getvalue()


def load_target_data(s3, bucket: str, dataset_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = load_s3_parquet(s3, bucket, f"{dataset_prefix.strip('/')}/train.parquet")
    test = load_s3_parquet(s3, bucket, f"{dataset_prefix.strip('/')}/test.parquet")
    if train is None:
        raise FileNotFoundError(f"s3://{bucket}/{dataset_prefix}/train.parquet")
    if test is None:
        raise FileNotFoundError(f"s3://{bucket}/{dataset_prefix}/test.parquet")
    return train, test


def select_stage1_features(
    common_columns: set[str],
    *,
    hmm_feature_prefix: str,
    require_hmm_features: bool,
) -> list[str]:
    feature_columns = selected_experiment_features(common_columns)
    hmm_columns = sorted(
        column
        for column in common_columns
        if hmm_feature_prefix and column.startswith(f"{hmm_feature_prefix}_")
    )
    if require_hmm_features and not hmm_columns:
        raise ValueError(
            f"No HMM features found with prefix {hmm_feature_prefix!r}. "
            "Use a with_price_hmm_n4 dataset prefix or pass --allow-missing-hmm."
        )
    return available_features(common_columns, [*feature_columns, *hmm_columns])


def prepare_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    *,
    hmm_feature_prefix: str,
    require_hmm_features: bool,
) -> PreparedData:
    common_columns = set(train.columns).intersection(test.columns)
    feature_columns = select_stage1_features(
        common_columns,
        hmm_feature_prefix=hmm_feature_prefix,
        require_hmm_features=require_hmm_features,
    )
    if not feature_columns:
        raise ValueError("No known experiment features are available in train/test")
    if target_column not in train.columns or target_column not in test.columns:
        raise ValueError(f"Target column is missing: {target_column}")

    y_train = pd.to_numeric(train[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_test = pd.to_numeric(test[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    train_mask = y_train.notna()
    test_mask = y_test.notna()

    X_train = train.loc[train_mask, feature_columns].replace([np.inf, -np.inf], np.nan)
    X_test = test.loc[test_mask, feature_columns].replace([np.inf, -np.inf], np.nan)
    medians = X_train.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_train = X_train.fillna(medians).astype(np.float32)
    X_test = X_test.fillna(medians).astype(np.float32)

    train_timestamps = train.loc[train_mask, "timestamp"].to_numpy() if "timestamp" in train.columns else None
    test_timestamps = test.loc[test_mask, "timestamp"].to_numpy() if "timestamp" in test.columns else None
    return PreparedData(
        feature_columns=feature_columns,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.loc[train_mask].to_numpy(dtype=np.float64),
        y_test=y_test.loc[test_mask].to_numpy(dtype=np.float64),
        train_timestamps=train_timestamps,
        test_timestamps=test_timestamps,
    )


def catboost_grid() -> list[dict[str, Any]]:
    return [
        {
            "depth": depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": l2_leaf_reg,
            "bootstrap_type": "Bayesian",
            "iterations": 3000,
        }
        for depth in (4, 6, 8, 10)
        for learning_rate in (0.03, 0.05, 0.10)
        for l2_leaf_reg in (3, 10, 30)
    ]


def lightgbm_grid() -> list[dict[str, Any]]:
    return [
        {
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "n_estimators": 3000,
        }
        for num_leaves in (31, 63, 127)
        for learning_rate in (0.03, 0.05, 0.10)
        for max_depth in (6, 10, -1)
    ]


def build_jobs(include_lightgbm_quantile: bool) -> list[SearchJob]:
    jobs: list[SearchJob] = []
    for index, params in enumerate(catboost_grid(), start=1):
        suffix = f"d{params['depth']}_lr{params['learning_rate']}_l2{params['l2_leaf_reg']}"
        jobs.append(SearchJob(f"catboost_rmse_{index:03d}_{suffix}", "catboost", "RMSE", params))
        jobs.append(
            SearchJob(
                f"catboost_multiquantile_{index:03d}_{suffix}",
                "catboost",
                "MultiQuantile",
                params,
            )
        )
        jobs.append(
            SearchJob(
                f"catboost_uncertainty_{index:03d}_{suffix}",
                "catboost",
                "RMSEWithUncertainty",
                params,
            )
        )

    for index, params in enumerate(lightgbm_grid(), start=1):
        suffix = (
            f"leaves{params['num_leaves']}_lr{params['learning_rate']}_"
            f"depth{params['max_depth']}"
        )
        jobs.append(SearchJob(f"lightgbm_rmse_{index:03d}_{suffix}", "lightgbm", "RMSE", params))
        if include_lightgbm_quantile:
            for alpha in QUANTILE_ALPHAS:
                jobs.append(
                    SearchJob(
                        f"lightgbm_quantile_a{alpha:g}_{index:03d}_{suffix}",
                        "lightgbm",
                        "Quantile",
                        params,
                        alpha=alpha,
                    )
                )
    return jobs


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0) & (y_pred != 0)
    if valid.sum() == 0:
        return np.nan
    return float((np.sign(y_true[valid]) == np.sign(y_pred[valid])).mean())


def threshold_direction_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> tuple[float, float, int]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) >= threshold)
    count = int(mask.sum())
    coverage = float(mask.mean()) if len(mask) else np.nan
    if count == 0:
        return np.nan, coverage, 0
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean()), coverage, count


def safe_corr(func, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 3 or np.nanstd(y_true[valid]) == 0 or np.nanstd(y_pred[valid]) == 0:
        return np.nan
    return float(func(y_true[valid], y_pred[valid])[0])


def pinball_loss(y_true: np.ndarray, y_quantile: np.ndarray, alpha: float) -> float:
    delta = y_true - y_quantile
    return float(np.mean(np.maximum(alpha * delta, (alpha - 1.0) * delta)))


def normal_nll(y_true: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.maximum(sigma, 1e-12)
    return float(np.mean(0.5 * np.log(2.0 * math.pi * sigma**2) + ((y_true - mean) ** 2) / (2.0 * sigma**2)))


def build_point_metrics(
    *,
    job: SearchJob,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fit_time: float,
    predict_time: float,
    feature_count: int,
    model_size_mb: float,
    direction_threshold: float,
    quantiles: dict[float, np.ndarray] | None = None,
    sigma: np.ndarray | None = None,
) -> dict[str, Any]:
    residual = y_true - y_pred
    da_025, da_025_coverage, da_025_count = threshold_direction_accuracy(
        y_true,
        y_pred,
        direction_threshold,
    )
    row: dict[str, Any] = {
        "job_id": job.job_id,
        "model_family": job.model_family,
        "loss_function": job.loss_function,
        "feature_set": "selected_experiment_features",
        "n_features": feature_count,
        "train_time": fit_time,
        "predict_time": predict_time,
        "model_size_mb": model_size_mb,
        "alpha": job.alpha,
        "params": json.dumps(job.params, ensure_ascii=False, sort_keys=True),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Direction_Accuracy": direction_accuracy(y_true, y_pred),
        "Direction_Accuracy_0.25%": da_025,
        "DA_025_coverage": da_025_coverage,
        "DA_025_count": da_025_count,
        "Pearson": safe_corr(pearsonr, y_true, y_pred),
        "Spearman": safe_corr(spearmanr, y_true, y_pred),
        "error_mean": float(np.mean(residual)),
        "error_std": float(np.std(residual)),
        "error_skew": float(skew(residual, nan_policy="omit")),
        "error_kurtosis": float(kurtosis(residual, nan_policy="omit")),
        "PinballLoss_05": np.nan,
        "PinballLoss_25": np.nan,
        "PinballLoss_50": np.nan,
        "PinballLoss_75": np.nan,
        "PinballLoss_95": np.nan,
        "Coverage90": np.nan,
        "Coverage95": np.nan,
        "IntervalWidth90": np.nan,
        "IntervalWidth95": np.nan,
        "NLL": np.nan,
        "CRPS": np.nan,
    }
    if quantiles:
        for alpha, suffix in ((0.05, "05"), (0.25, "25"), (0.50, "50"), (0.75, "75"), (0.95, "95")):
            if alpha in quantiles:
                row[f"PinballLoss_{suffix}"] = pinball_loss(y_true, quantiles[alpha], alpha)
        if 0.05 in quantiles and 0.95 in quantiles:
            lower = quantiles[0.05]
            upper = quantiles[0.95]
            row["Coverage90"] = float(((y_true >= lower) & (y_true <= upper)).mean())
            row["IntervalWidth90"] = float(np.mean(upper - lower))
    if sigma is not None:
        row["NLL"] = normal_nll(y_true, y_pred, sigma)
    return row


def fit_catboost(job: SearchJob, data: PreparedData, threads_per_model: int) -> tuple[Any, np.ndarray, dict[float, np.ndarray] | None, np.ndarray | None, float, float]:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("catboost is not installed. Run: pip install catboost") from exc

    params = {
        **job.params,
        "random_seed": 42,
        "thread_count": threads_per_model,
        "verbose": False,
        "allow_writing_files": False,
    }
    if job.loss_function == "MultiQuantile":
        params["loss_function"] = "MultiQuantile:alpha=" + ",".join(str(alpha) for alpha in QUANTILE_ALPHAS)
        params["eval_metric"] = "MultiQuantile"
    else:
        params["loss_function"] = job.loss_function

    model = CatBoostRegressor(**params)
    started = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    fit_time = time.perf_counter() - started

    started = time.perf_counter()
    prediction = np.asarray(model.predict(data.X_test))
    predict_time = time.perf_counter() - started

    quantiles = None
    sigma = None
    if job.loss_function == "MultiQuantile":
        if prediction.ndim != 2:
            raise RuntimeError("CatBoost MultiQuantile returned an unexpected prediction shape")
        quantiles = {alpha: prediction[:, idx] for idx, alpha in enumerate(QUANTILE_ALPHAS)}
        y_pred = quantiles[0.50]
    elif job.loss_function == "RMSEWithUncertainty" and prediction.ndim == 2:
        y_pred = prediction[:, 0]
        if prediction.shape[1] > 1:
            sigma = np.sqrt(np.maximum(prediction[:, 1], 1e-12))
    else:
        y_pred = prediction.reshape(-1)
    return model, y_pred, quantiles, sigma, fit_time, predict_time


def fit_lightgbm(job: SearchJob, data: PreparedData, threads_per_model: int) -> tuple[Any, np.ndarray, dict[float, np.ndarray] | None, np.ndarray | None, float, float]:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError("lightgbm is not installed. Run: pip install lightgbm") from exc

    objective = "quantile" if job.loss_function == "Quantile" else "regression"
    params = {
        **job.params,
        "objective": objective,
        "random_state": 42,
        "n_jobs": threads_per_model,
        "verbosity": -1,
    }
    if job.alpha is not None:
        params["alpha"] = job.alpha

    model = LGBMRegressor(**params)
    started = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    fit_time = time.perf_counter() - started

    started = time.perf_counter()
    y_pred = np.asarray(model.predict(data.X_test)).reshape(-1)
    predict_time = time.perf_counter() - started
    quantiles = {job.alpha: y_pred} if job.alpha is not None else None
    return model, y_pred, quantiles, None, fit_time, predict_time


def feature_importance_frame(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    if hasattr(model, "get_feature_importance"):
        importance = model.get_feature_importance()
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": feature_columns, "importance": np.asarray(importance, dtype=float)})


def predictions_frame(
    data: PreparedData,
    y_pred: np.ndarray,
    quantiles: dict[float, np.ndarray] | None,
    sigma: np.ndarray | None,
) -> pd.DataFrame:
    frame = pd.DataFrame({"y_true": data.y_test, "y_pred": y_pred})
    if data.test_timestamps is not None:
        frame.insert(0, "timestamp", data.test_timestamps)
    if quantiles:
        for alpha, values in sorted(quantiles.items()):
            frame[f"q{int(alpha * 100):02d}"] = values
    if sigma is not None:
        frame["sigma"] = sigma
    return frame


def run_one_job(
    *,
    job: SearchJob,
    data: PreparedData,
    bucket: str,
    output_prefix: str,
    direction_threshold: float,
    threads_per_model: int,
    overwrite: bool,
) -> dict[str, Any]:
    s3 = make_s3_client()
    job_prefix = f"{output_prefix}/jobs/{job.job_id}"
    metrics_key = f"{job_prefix}/metrics.json"
    if not overwrite and object_exists(s3, bucket, metrics_key):
        return {"job_id": job.job_id, "status": "skipped", "metrics_key": metrics_key}

    started = time.perf_counter()
    if job.model_family == "catboost":
        model, y_pred, quantiles, sigma, fit_time, predict_time = fit_catboost(job, data, threads_per_model)
    elif job.model_family == "lightgbm":
        model, y_pred, quantiles, sigma, fit_time, predict_time = fit_lightgbm(job, data, threads_per_model)
    else:
        raise ValueError(f"Unknown model family: {job.model_family}")

    model_payload = serialize_model(model)
    metrics = build_point_metrics(
        job=job,
        y_true=data.y_test,
        y_pred=y_pred,
        fit_time=fit_time,
        predict_time=predict_time,
        feature_count=len(data.feature_columns),
        model_size_mb=len(model_payload) / 1024**2,
        direction_threshold=direction_threshold,
        quantiles=quantiles,
        sigma=sigma,
    )
    metrics.update(
        {
            "status": "completed",
            "elapsed_sec": time.perf_counter() - started,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_key": f"{job_prefix}/model.joblib",
            "predictions_key": f"{job_prefix}/predictions.parquet",
            "feature_importance_key": f"{job_prefix}/feature_importance.parquet",
            "metrics_key": metrics_key,
        }
    )

    upload_bytes(s3, bucket, metrics["model_key"], model_payload, "application/octet-stream")
    upload_parquet(s3, bucket, metrics["predictions_key"], predictions_frame(data, y_pred, quantiles, sigma))
    upload_parquet(s3, bucket, metrics["feature_importance_key"], feature_importance_frame(model, data.feature_columns))
    upload_json(s3, bucket, f"{job_prefix}/metadata.json", {"job": job.__dict__, "features": data.feature_columns})
    upload_json(s3, bucket, metrics_key, metrics)
    return metrics


def read_json_object(s3, bucket: str, key: str) -> dict[str, Any]:
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8"))


def collect_metrics(s3, bucket: str, output_prefix: str) -> pd.DataFrame:
    paginator = s3.get_paginator("list_objects_v2")
    records = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{output_prefix}/jobs/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/metrics.json"):
                records.append(read_json_object(s3, bucket, key))
    return pd.DataFrame(records)


def build_leaderboards(results: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if results.empty:
        return results
    ranked = results.copy()
    sort_columns = ["RMSE", "MAE", "Direction_Accuracy_0.25%", "job_id"]
    ranked = ranked.sort_values(sort_columns, ascending=[True, True, False, True])
    ranked["global_rank"] = np.arange(1, len(ranked) + 1)
    ranked["family_loss_rank"] = (
        ranked.groupby(["model_family", "loss_function"], dropna=False)
        .cumcount()
        .astype(int)
        + 1
    )
    return ranked.loc[ranked["family_loss_rank"].le(top_n)].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 architecture search for target models.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--horizon", type=int, default=20, choices=sorted(TARGET_SPECS))
    parser.add_argument("--dataset-prefix", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--results-subdir", default=DEFAULT_RESULTS_SUBDIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-parallel-models", type=int, default=6)
    parser.add_argument("--threads-per-model", type=int, default=4)
    parser.add_argument("--direction-threshold", type=float, default=DEFAULT_DIRECTION_THRESHOLD)
    parser.add_argument("--hmm-feature-prefix", default=DEFAULT_HMM_FEATURE_PREFIX)
    parser.add_argument(
        "--allow-missing-hmm",
        action="store_true",
        help="Allow training even if the selected dataset has no HMM probability features.",
    )
    parser.add_argument("--include-lightgbm-quantile", action="store_true")
    parser.add_argument("--limit-jobs", type=int, default=None, help="Smoke-test limit.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build the queue and config without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    run_id = args.run_id or make_run_id()
    default_dataset_prefix, default_target = TARGET_SPECS[args.horizon]
    dataset_prefix = args.dataset_prefix or default_dataset_prefix
    target_column = args.target or default_target
    output_prefix = f"{dataset_prefix.strip('/')}/{args.results_subdir.strip('/')}/{run_id}"
    latest_prefix = f"{dataset_prefix.strip('/')}/{args.results_subdir.strip('/')}/latest"

    jobs = build_jobs(args.include_lightgbm_quantile)
    if args.limit_jobs is not None:
        jobs = jobs[: args.limit_jobs]
    run_config = {
        "run_id": run_id,
        "horizon": args.horizon,
        "dataset_prefix": dataset_prefix,
        "target_column": target_column,
        "results_subdir": args.results_subdir,
        "output_prefix": f"s3://{args.bucket}/{output_prefix}/",
        "max_parallel_models": args.max_parallel_models,
        "threads_per_model": args.threads_per_model,
        "reserved_vcpu_hint": max(0, 32 - args.max_parallel_models * args.threads_per_model),
        "direction_threshold": args.direction_threshold,
        "hmm_feature_prefix": args.hmm_feature_prefix,
        "require_hmm_features": not args.allow_missing_hmm,
        "include_lightgbm_quantile": args.include_lightgbm_quantile,
        "jobs": len(jobs),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    logger.info("run_prefix=s3://%s/%s/", args.bucket, output_prefix)
    logger.info("jobs=%d max_parallel_models=%d threads_per_model=%d", len(jobs), args.max_parallel_models, args.threads_per_model)
    if args.dry_run:
        counts = pd.DataFrame([job.__dict__ for job in jobs]).groupby(
            ["model_family", "loss_function"],
            dropna=False,
        ).size()
        for (model_family, loss_function), count in counts.items():
            logger.info("queue %s %s jobs=%d", model_family, loss_function, count)
        logger.info("dry-run enabled; skipping S3 upload and training")
        return

    s3 = make_s3_client()
    upload_json(s3, args.bucket, f"{output_prefix}/run_config.json", run_config)
    upload_json(s3, args.bucket, f"{latest_prefix}/run_config.json", run_config)
    upload_parquet(s3, args.bucket, f"{output_prefix}/job_queue.parquet", pd.DataFrame([job.__dict__ for job in jobs]))
    logger.info("loading train/test from s3://%s/%s", args.bucket, dataset_prefix)
    train, test = load_target_data(s3, args.bucket, dataset_prefix)
    data = prepare_data(
        train,
        test,
        target_column,
        hmm_feature_prefix=args.hmm_feature_prefix,
        require_hmm_features=not args.allow_missing_hmm,
    )
    logger.info(
        "prepared train_rows=%d test_rows=%d features=%d",
        len(data.y_train),
        len(data.y_test),
        len(data.feature_columns),
    )

    completed = 0
    skipped = 0
    failed = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.max_parallel_models) as executor:
        futures = {
            executor.submit(
                run_one_job,
                job=job,
                data=data,
                bucket=args.bucket,
                output_prefix=output_prefix,
                direction_threshold=args.direction_threshold,
                threads_per_model=args.threads_per_model,
                overwrite=args.overwrite,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                if result.get("status") == "skipped":
                    skipped += 1
                    logger.info("SKIP %s", job.job_id)
                else:
                    completed += 1
                    logger.info(
                        "DONE %s RMSE=%.8f DA_025=%.6f elapsed=%.1fs",
                        job.job_id,
                        result["RMSE"],
                        result["Direction_Accuracy_0.25%"],
                        result["elapsed_sec"],
                    )
            except Exception as exc:
                failed.append({"job_id": job.job_id, "error": str(exc)})
                logger.exception("FAILED %s", job.job_id)

    results = collect_metrics(s3, args.bucket, output_prefix)
    leaderboard = build_leaderboards(results, top_n=10)
    upload_parquet(s3, args.bucket, f"{output_prefix}/stage1_results.parquet", results)
    upload_csv(s3, args.bucket, f"{output_prefix}/stage1_results.csv", results)
    upload_parquet(s3, args.bucket, f"{output_prefix}/leaderboard_top10.parquet", leaderboard)
    upload_csv(s3, args.bucket, f"{output_prefix}/leaderboard_top10.csv", leaderboard)
    upload_parquet(s3, args.bucket, f"{latest_prefix}/stage1_results.parquet", results)
    upload_parquet(s3, args.bucket, f"{latest_prefix}/leaderboard_top10.parquet", leaderboard)
    upload_json(
        s3,
        args.bucket,
        f"{output_prefix}/run_summary.json",
        {
            **run_config,
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "elapsed_sec": time.perf_counter() - started,
            "result_rows": len(results),
        },
    )
    logger.info("completed=%d skipped=%d failed=%d elapsed_sec=%.1f", completed, skipped, len(failed), time.perf_counter() - started)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
