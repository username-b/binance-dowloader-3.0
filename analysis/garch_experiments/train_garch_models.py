from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning
from scipy.special import gammaln


DEFAULT_BUCKET = "binance-data-downloader"
DEFAULT_DATASET_PREFIX = "dataset_target_10"
DEFAULT_TARGET_COLUMN = "target_log_return_10m"
DEFAULT_TIMESTAMP_COLUMN = "timestamp"
DEFAULT_RESULTS_SUBDIR = "garch_experiments"


@dataclass(frozen=True)
class GarchSpec:
    name: str
    vol: str
    p: int = 1
    o: int = 0
    q: int = 1
    power: float = 2.0


MODEL_SPECS = [
    GarchSpec(name="garch_1_1", vol="GARCH", p=1, o=0, q=1),
    GarchSpec(name="egarch_1_1", vol="EGARCH", p=1, o=1, q=1),
    GarchSpec(name="gjr_garch_1_1", vol="GARCH", p=1, o=1, q=1),
    GarchSpec(name="aparch_1_1", vol="APARCH", p=1, o=1, q=1),
]


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "build_price_feature_day.py").exists():
            return candidate
    raise FileNotFoundError("Could not find project root with build_price_feature_day.py")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_price_feature_day import load_s3_parquet, make_s3_client


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("garch_experiments")


def upload_bytes(s3, bucket: str, key: str, payload: bytes, content_type: str) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)


def upload_dataframe_parquet(s3, bucket: str, key: str, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, engine="pyarrow", compression="zstd")
    upload_bytes(s3, bucket, key, buffer.getvalue(), "application/vnd.apache.parquet")


def upload_dataframe_csv(s3, bucket: str, key: str, frame: pd.DataFrame) -> None:
    upload_bytes(s3, bucket, key, frame.to_csv(index=False).encode("utf-8-sig"), "text/csv; charset=utf-8")


def load_train_test(
    *,
    s3,
    bucket: str,
    dataset_prefix: str,
    train_path: Path | None,
    test_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if train_path or test_path:
        if train_path is None or test_path is None:
            raise ValueError("--train-path and --test-path must be provided together")
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)
        return train, test, f"local:{train_path},{test_path}"

    train_key = f"{dataset_prefix.strip('/')}/train.parquet"
    test_key = f"{dataset_prefix.strip('/')}/test.parquet"
    train = load_s3_parquet(s3, bucket, train_key)
    test = load_s3_parquet(s3, bucket, test_key)
    if train is None:
        raise FileNotFoundError(f"s3://{bucket}/{train_key}")
    if test is None:
        raise FileNotFoundError(f"s3://{bucket}/{test_key}")
    return train, test, f"s3://{bucket}/{dataset_prefix.strip('/')}/"


def clean_returns(
    frame: pd.DataFrame,
    *,
    target_column: str,
    timestamp_column: str,
) -> tuple[pd.Series, pd.Series | None]:
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column: {target_column}")

    values = pd.to_numeric(frame[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = values.notna()
    returns = values.loc[mask].astype("float64").reset_index(drop=True)
    timestamps = None
    if timestamp_column in frame.columns:
        timestamps = frame.loc[mask, timestamp_column].reset_index(drop=True)
    return returns, timestamps


def standardized_student_logpdf(z: np.ndarray, nu: float) -> np.ndarray:
    if not np.isfinite(nu) or nu <= 2.0:
        return np.full_like(z, np.nan, dtype=np.float64)
    c = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * math.log(math.pi * (nu - 2.0))
    return c - ((nu + 1.0) / 2.0) * np.log1p(np.square(z) / (nu - 2.0))


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return np.nan
    return float(values[valid].mean())


def fit_one_model(
    spec: GarchSpec,
    *,
    train_returns: pd.Series,
    test_returns: pd.Series,
    test_timestamps: pd.Series | None,
    mean: str,
    dist: str,
    scale: float,
    maxiter: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    started = time.perf_counter()
    train_scaled = train_returns.to_numpy(dtype=np.float64) * scale
    test_scaled = test_returns.to_numpy(dtype=np.float64) * scale
    combined = pd.Series(np.concatenate([train_scaled, test_scaled]), dtype="float64")

    model = arch_model(
        combined,
        mean=mean,
        vol=spec.vol,
        p=spec.p,
        o=spec.o,
        q=spec.q,
        power=spec.power,
        dist=dist,
        rescale=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        result = model.fit(
            last_obs=len(train_scaled) - 1,
            disp="off",
            options={"maxiter": maxiter},
        )

    forecast = result.forecast(horizon=1, start=len(train_scaled), reindex=False)
    mean_scaled = forecast.mean.iloc[: len(test_scaled), 0].to_numpy(dtype=np.float64)
    variance_scaled = forecast.variance.iloc[: len(test_scaled), 0].to_numpy(dtype=np.float64)
    variance_scaled = np.maximum(variance_scaled, np.finfo(np.float64).tiny)
    sigma_scaled = np.sqrt(variance_scaled)
    residual_scaled = test_scaled - mean_scaled
    standardized_residual = residual_scaled / sigma_scaled

    nu = float(result.params.get("nu", np.nan))
    logpdf = standardized_student_logpdf(standardized_residual, nu) - np.log(sigma_scaled)
    realized_variance_scaled = np.square(residual_scaled)
    qlike = np.log(variance_scaled) + realized_variance_scaled / variance_scaled

    fit_time = time.perf_counter() - started
    converged = int(getattr(result, "convergence_flag", 1)) == 0
    warning_messages = sorted({str(item.message) for item in caught})

    row: dict[str, Any] = {
        "model_name": spec.name,
        "vol": spec.vol,
        "p": spec.p,
        "o": spec.o,
        "q": spec.q,
        "power": spec.power,
        "mean": mean,
        "dist": dist,
        "scale": scale,
        "train_size": int(len(train_scaled)),
        "test_size": int(len(test_scaled)),
        "fit_time": fit_time,
        "converged": converged,
        "convergence_flag": int(getattr(result, "convergence_flag", -1)),
        "loglikelihood": float(result.loglikelihood),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "nu": nu,
        "omega": float(result.params.get("omega", np.nan)),
        "alpha_1": float(result.params.get("alpha[1]", np.nan)),
        "gamma_1": float(result.params.get("gamma[1]", np.nan)),
        "beta_1": float(result.params.get("beta[1]", np.nan)),
        "delta": float(result.params.get("delta", np.nan)),
        "test_nll": -safe_mean(logpdf),
        "test_qlike": safe_mean(qlike),
        "variance_mse": safe_mean(np.square((variance_scaled - realized_variance_scaled) / (scale**2))),
        "volatility_mae": safe_mean(np.abs((sigma_scaled - np.abs(residual_scaled)) / scale)),
        "std_resid_mean": safe_mean(standardized_residual),
        "std_resid_std": float(np.nanstd(standardized_residual)),
        "warnings": " | ".join(warning_messages),
    }

    predictions = pd.DataFrame(
        {
            "row_id": np.arange(len(test_scaled), dtype=np.int64),
            "model_name": spec.name,
            "y_true": test_scaled / scale,
            "mean_forecast": mean_scaled / scale,
            "variance_forecast": variance_scaled / (scale**2),
            "volatility_forecast": sigma_scaled / scale,
            "standardized_residual": standardized_residual,
            "loglikelihood": logpdf,
        }
    )
    if test_timestamps is not None:
        predictions.insert(0, "timestamp", test_timestamps.to_numpy())
    return row, predictions


def run_garch_experiments(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str = DEFAULT_TARGET_COLUMN,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
    mean: str = "Constant",
    dist: str = "studentst",
    scale: float = 100.0,
    maxiter: int = 1000,
    specs: list[GarchSpec] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_returns, _ = clean_returns(train, target_column=target_column, timestamp_column=timestamp_column)
    test_returns, test_timestamps = clean_returns(test, target_column=target_column, timestamp_column=timestamp_column)
    if len(train_returns) < 100:
        raise ValueError(f"Need at least 100 non-null training returns, got {len(train_returns)}")
    if test_returns.empty:
        raise ValueError("Test split has no non-null returns")

    rows = []
    prediction_frames = []
    for spec in specs or MODEL_SPECS:
        if logger:
            logger.info("START model=%s vol=%s p=%d o=%d q=%d", spec.name, spec.vol, spec.p, spec.o, spec.q)
        row, predictions = fit_one_model(
            spec,
            train_returns=train_returns,
            test_returns=test_returns,
            test_timestamps=test_timestamps,
            mean=mean,
            dist=dist,
            scale=scale,
            maxiter=maxiter,
        )
        rows.append(row)
        prediction_frames.append(predictions)
        if logger:
            logger.info(
                "DONE model=%s aic=%.3f bic=%.3f test_nll=%.6f qlike=%.6f nu=%.3f fit_sec=%.1f",
                row["model_name"],
                row["aic"],
                row["bic"],
                row["test_nll"],
                row["test_qlike"],
                row["nu"],
                row["fit_time"],
            )

    results = pd.DataFrame(rows).sort_values(
        ["test_nll", "test_qlike", "bic", "model_name"],
        na_position="last",
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return results, predictions


def save_local_outputs(
    output_dir: Path,
    *,
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output_dir / "garch_results.parquet", index=False)
    results.to_csv(output_dir / "garch_results.csv", index=False, encoding="utf-8-sig")
    predictions.to_parquet(output_dir / "garch_predictions.parquet", index=False)
    (output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def save_s3_outputs(
    s3,
    bucket: str,
    dataset_prefix: str,
    results_subdir: str,
    run_id: str,
    *,
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, str]:
    output_prefix = f"{dataset_prefix.strip('/')}/{results_subdir.strip('/')}/{run_id}"
    latest_prefix = f"{dataset_prefix.strip('/')}/{results_subdir.strip('/')}/latest"
    artifacts = {
        f"{output_prefix}/garch_results.parquet": ("parquet", results),
        f"{output_prefix}/garch_results.csv": ("csv", results),
        f"{output_prefix}/garch_predictions.parquet": ("parquet", predictions),
        f"{latest_prefix}/garch_results.parquet": ("parquet", results),
        f"{latest_prefix}/garch_predictions.parquet": ("parquet", predictions),
    }
    for key, (kind, frame) in artifacts.items():
        if kind == "parquet":
            upload_dataframe_parquet(s3, bucket, key, frame)
        else:
            upload_dataframe_csv(s3, bucket, key, frame)

    payload = json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8")
    upload_bytes(s3, bucket, f"{output_prefix}/run_config.json", payload, "application/json")
    upload_bytes(s3, bucket, f"{latest_prefix}/run_config.json", payload, "application/json")
    return {
        "run_prefix": f"s3://{bucket}/{output_prefix}/",
        "latest_prefix": f"s3://{bucket}/{latest_prefix}/",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Student-t GARCH family models for target returns.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--dataset-prefix", default=DEFAULT_DATASET_PREFIX)
    parser.add_argument("--target", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--timestamp-column", default=DEFAULT_TIMESTAMP_COLUMN)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--test-path", type=Path, default=None)
    parser.add_argument("--results-subdir", default=DEFAULT_RESULTS_SUBDIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mean", default="Constant", choices=["Constant", "Zero"])
    parser.add_argument("--dist", default="studentst", choices=["studentst", "t"])
    parser.add_argument("--scale", type=float, default=100.0)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", help="Train and print results without S3 upload.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    run_id = args.run_id or make_run_id()
    s3 = None if args.train_path else make_s3_client()

    logger.info("project_root=%s", PROJECT_ROOT)
    logger.info("run_id=%s", run_id)
    train, test, source = load_train_test(
        s3=s3,
        bucket=args.bucket,
        dataset_prefix=args.dataset_prefix,
        train_path=args.train_path,
        test_path=args.test_path,
    )
    logger.info("source=%s train_rows=%d test_rows=%d target=%s", source, len(train), len(test), args.target)

    results, predictions = run_garch_experiments(
        train,
        test,
        target_column=args.target,
        timestamp_column=args.timestamp_column,
        mean=args.mean,
        dist=args.dist,
        scale=args.scale,
        maxiter=args.maxiter,
        logger=logger,
    )
    logger.info("leaderboard:\n%s", results[["model_name", "test_nll", "test_qlike", "bic", "nu", "converged"]].to_string(index=False))

    config = {
        "run_id": run_id,
        "source": source,
        "bucket": args.bucket,
        "dataset_prefix": args.dataset_prefix,
        "target": args.target,
        "timestamp_column": args.timestamp_column,
        "results_subdir": args.results_subdir,
        "mean": args.mean,
        "dist": args.dist,
        "scale": args.scale,
        "maxiter": args.maxiter,
        "models": [asdict(spec) for spec in MODEL_SPECS],
        "train_rows": len(train),
        "test_rows": len(test),
    }

    if args.output_dir:
        save_local_outputs(args.output_dir, results=results, predictions=predictions, config=config)
        logger.info("local_output=%s", args.output_dir)

    if args.dry_run or args.train_path:
        logger.info("skipping S3 upload")
        return

    locations = save_s3_outputs(
        s3,
        args.bucket,
        args.dataset_prefix,
        args.results_subdir,
        run_id,
        results=results,
        predictions=predictions,
        config=config,
    )
    logger.info("run_prefix=%s", locations["run_prefix"])
    logger.info("latest_prefix=%s", locations["latest_prefix"])


if __name__ == "__main__":
    main()
