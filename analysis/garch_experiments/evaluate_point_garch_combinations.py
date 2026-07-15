from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, t


DEFAULT_BUCKET = "binance-data-downloader"
DEFAULT_RESULTS_PREFIX = "stage1_model_benchmarks"
DEFAULT_STAGE1_SUBDIR = "stage1_architecture_search"
DEFAULT_GARCH_SUBDIR = "garch_experiments"
DEFAULT_OUTPUT_SUBDIR = "point_garch_probabilistic_eval"
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

from build_price_feature_day import make_s3_client
from analysis.garch_experiments.train_garch_models import standardized_student_logpdf


def read_bytes(s3: Any, bucket: str, key: str) -> bytes:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def read_json(s3: Any, bucket: str, key: str) -> dict[str, Any]:
    return json.loads(read_bytes(s3, bucket, key).decode("utf-8"))


def read_parquet(s3: Any, bucket: str, key: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(read_bytes(s3, bucket, key)))


def upload_bytes(s3: Any, bucket: str, key: str, payload: bytes, content_type: str) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)


def upload_parquet(s3: Any, bucket: str, key: str, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, engine="pyarrow", compression="zstd")
    upload_bytes(s3, bucket, key, buffer.getvalue(), "application/vnd.apache.parquet")


def upload_csv(s3: Any, bucket: str, key: str, frame: pd.DataFrame) -> None:
    upload_bytes(s3, bucket, key, frame.to_csv(index=False).encode("utf-8-sig"), "text/csv; charset=utf-8")


def resolve_stage1_prefix(
    *,
    results_prefix: str,
    suite: str,
    horizon: int,
    stage1_run_id: str,
) -> str:
    return f"{results_prefix.strip('/')}/{suite}/horizon_{horizon}/{stage1_run_id.strip('/')}"


def resolve_garch_prefix(
    *,
    dataset_prefix: str,
    garch_subdir: str,
    garch_run_id: str,
) -> str:
    return f"{dataset_prefix.strip('/')}/{garch_subdir.strip('/')}/{garch_run_id.strip('/')}"


def select_top_point_models(
    results: pd.DataFrame,
    *,
    top_n: int,
    metric: str,
) -> pd.DataFrame:
    if results.empty:
        raise ValueError("Point model results table is empty")
    if metric not in results.columns:
        raise ValueError(f"Point model results do not contain metric: {metric}")
    if "job_id" not in results.columns:
        raise ValueError("Point model results must contain job_id")
    frame = results.copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    return frame.sort_values([metric, "job_id"], na_position="last").head(top_n)


def best_garch_model(results: pd.DataFrame, metric: str = "test_nll") -> str:
    if results.empty:
        raise ValueError("GARCH results table is empty")
    if metric not in results.columns:
        raise ValueError(f"GARCH results do not contain metric: {metric}")
    ranked = results.copy()
    ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
    return str(ranked.sort_values([metric, "model_name"], na_position="last").iloc[0]["model_name"])


def parse_requested_models(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    models = []
    for value in values:
        models.extend(part.strip() for part in str(value).split(",") if part.strip())
    return models or None


def select_garch_models(
    results: pd.DataFrame,
    *,
    requested_models: list[str] | None = None,
    metric: str = "test_nll",
    selection: str = "all",
) -> list[str]:
    if results.empty:
        raise ValueError("GARCH results table is empty")
    if "model_name" not in results.columns:
        raise ValueError("GARCH results must contain model_name")

    available = set(results["model_name"].astype(str))
    if requested_models:
        missing = [model for model in requested_models if model not in available]
        if missing:
            raise ValueError(f"Requested GARCH models are absent from results: {missing}")
        return requested_models

    if selection == "best":
        return [best_garch_model(results, metric=metric)]
    if selection != "all":
        raise ValueError(f"Unsupported GARCH selection: {selection}")

    ranked = results.copy()
    if metric in ranked.columns:
        ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
        ranked = ranked.sort_values([metric, "model_name"], na_position="last")
    else:
        ranked = ranked.sort_values("model_name")
    return ranked["model_name"].astype(str).drop_duplicates().tolist()


def align_point_and_garch(point: pd.DataFrame, garch: pd.DataFrame) -> pd.DataFrame:
    point_columns = ["y_true", "y_pred"]
    missing = [column for column in point_columns if column not in point.columns]
    if missing:
        raise ValueError(f"Point predictions are missing columns: {missing}")
    if "volatility_forecast" not in garch.columns:
        raise ValueError("GARCH predictions must contain volatility_forecast")

    point_frame = point.copy()
    garch_frame = garch.copy()
    if "timestamp" in point_frame.columns and "timestamp" in garch_frame.columns:
        point_frame["timestamp"] = pd.to_datetime(point_frame["timestamp"], utc=True)
        garch_frame["timestamp"] = pd.to_datetime(garch_frame["timestamp"], utc=True)
        merged = point_frame.merge(
            garch_frame[["timestamp", "model_name", "volatility_forecast", "variance_forecast"]],
            on="timestamp",
            how="inner",
        )
    else:
        point_frame = point_frame.reset_index(drop=True).assign(row_id=lambda frame: np.arange(len(frame)))
        garch_frame = garch_frame.reset_index(drop=True).assign(row_id=lambda frame: np.arange(len(frame)))
        merged = point_frame.merge(
            garch_frame[["row_id", "model_name", "volatility_forecast", "variance_forecast"]],
            on="row_id",
            how="inner",
        )

    if merged.empty:
        raise ValueError("Point and GARCH predictions have no overlapping rows")
    return merged


def pinball_loss(y_true: np.ndarray, y_quantile: np.ndarray, alpha: float) -> float:
    delta = y_true - y_quantile
    return float(np.mean(np.maximum(alpha * delta, (alpha - 1.0) * delta)))


def quantile_crps_approx(y_true: np.ndarray, quantiles: dict[float, np.ndarray]) -> float:
    losses = np.array([2.0 * pinball_loss(y_true, quantiles[alpha], alpha) for alpha in QUANTILE_ALPHAS])
    alphas = np.array(QUANTILE_ALPHAS, dtype=np.float64)
    return float(np.trapezoid(losses, alphas))


def student_t_quantiles(mean: np.ndarray, sigma: np.ndarray, nu: float) -> dict[float, np.ndarray]:
    if not np.isfinite(nu) or nu <= 2.0:
        raise ValueError(f"Student-t df must be > 2, got {nu}")
    standardizer = math.sqrt((nu - 2.0) / nu)
    return {
        alpha: mean + sigma * standardizer * float(t.ppf(alpha, df=nu))
        for alpha in QUANTILE_ALPHAS
    }


def normal_quantiles(mean: np.ndarray, sigma: np.ndarray) -> dict[float, np.ndarray]:
    sigma = np.maximum(sigma, 1e-12)
    return {alpha: mean + sigma * float(norm.ppf(alpha)) for alpha in QUANTILE_ALPHAS}


def metric_row_from_distribution(
    *,
    model_name: str,
    point_model: str,
    garch_model: str | None,
    distribution: str,
    y_true: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    quantiles: dict[float, np.ndarray],
    nu: float | None = None,
) -> dict[str, Any]:
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    y_true = np.asarray(y_true, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    residual = y_true - mean

    if distribution == "studentst":
        if nu is None:
            raise ValueError("nu is required for Student-t evaluation")
        logpdf = standardized_student_logpdf(residual / sigma, nu) - np.log(sigma)
    elif distribution == "normal":
        logpdf = norm.logpdf(y_true, loc=mean, scale=sigma)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    lower90 = quantiles[0.05]
    upper90 = quantiles[0.95]
    return {
        "model_name": model_name,
        "point_model": point_model,
        "garch_model": garch_model,
        "distribution": distribution,
        "n_rows": int(len(y_true)),
        "mean_error": float(np.mean(residual)),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(math.sqrt(np.mean(np.square(residual)))),
        "NLL": float(-np.nanmean(logpdf)),
        "CRPS_approx": quantile_crps_approx(y_true, quantiles),
        "PinballLoss_05": pinball_loss(y_true, quantiles[0.05], 0.05),
        "PinballLoss_25": pinball_loss(y_true, quantiles[0.25], 0.25),
        "PinballLoss_50": pinball_loss(y_true, quantiles[0.50], 0.50),
        "PinballLoss_75": pinball_loss(y_true, quantiles[0.75], 0.75),
        "PinballLoss_95": pinball_loss(y_true, quantiles[0.95], 0.95),
        "Coverage90": float(((y_true >= lower90) & (y_true <= upper90)).mean()),
        "IntervalWidth90": float(np.mean(upper90 - lower90)),
        "sigma_mean": float(np.mean(sigma)),
        "sigma_median": float(np.median(sigma)),
        "nu": float(nu) if nu is not None else np.nan,
    }


def forecast_frame_from_distribution(
    *,
    point_frame: pd.DataFrame,
    merged: pd.DataFrame,
    point_model: str,
    garch_model: str,
    distribution: str,
    mean: np.ndarray,
    sigma: np.ndarray,
    quantiles: dict[float, np.ndarray],
    nu: float,
) -> pd.DataFrame:
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    y_true = pd.to_numeric(merged["y_true"], errors="coerce").to_numpy(dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    residual = y_true - mean
    logpdf = standardized_student_logpdf(residual / sigma, nu) - np.log(sigma)

    frame = pd.DataFrame(
        {
            "point_model": point_model,
            "garch_model": garch_model,
            "model_name": f"{point_model}+{garch_model}_{distribution}",
            "distribution": distribution,
            "y_true": y_true,
            "mean_forecast": mean,
            "sigma_forecast": sigma,
            "variance_forecast": np.square(sigma),
            "loglikelihood": logpdf,
            "nu": float(nu),
        }
    )
    for alpha, values in quantiles.items():
        frame[f"q{int(alpha * 100):02d}"] = values
    if "timestamp" in merged.columns:
        frame.insert(0, "timestamp", merged["timestamp"].to_numpy())
    elif "timestamp" in point_frame.columns:
        frame.insert(0, "timestamp", point_frame["timestamp"].to_numpy()[: len(frame)])
    if "row_id" in merged.columns:
        frame.insert(0, "row_id", merged["row_id"].to_numpy())
    else:
        frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    return frame


def build_point_garch_forecasts(
    *,
    point_predictions: dict[str, pd.DataFrame],
    garch_results: pd.DataFrame,
    garch_predictions: pd.DataFrame,
    garch_model_names: list[str] | None = None,
    garch_selection: str = "all",
    garch_metric: str = "test_nll",
) -> pd.DataFrame:
    selected_garch_models = select_garch_models(
        garch_results,
        requested_models=garch_model_names,
        metric=garch_metric,
        selection=garch_selection,
    )
    frames = []
    for selected_garch in selected_garch_models:
        garch_meta = garch_results.loc[garch_results["model_name"].astype(str).eq(selected_garch)]
        if garch_meta.empty:
            raise ValueError(f"GARCH model not found in results: {selected_garch}")
        nu = float(garch_meta.iloc[0].get("nu", np.nan))
        selected_garch_predictions = garch_predictions.loc[
            garch_predictions["model_name"].astype(str).eq(selected_garch)
        ].copy()
        if selected_garch_predictions.empty:
            raise ValueError(f"GARCH predictions not found for model: {selected_garch}")

        for point_name, point_frame in point_predictions.items():
            merged = align_point_and_garch(point_frame, selected_garch_predictions)
            mean = pd.to_numeric(merged["y_pred"], errors="coerce").to_numpy(dtype=np.float64)
            garch_sigma = pd.to_numeric(merged["volatility_forecast"], errors="coerce").to_numpy(dtype=np.float64)
            frames.append(
                forecast_frame_from_distribution(
                    point_frame=point_frame,
                    merged=merged,
                    point_model=point_name,
                    garch_model=selected_garch,
                    distribution="studentst",
                    mean=mean,
                    sigma=garch_sigma,
                    quantiles=student_t_quantiles(mean, garch_sigma, nu),
                    nu=nu,
                )
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def evaluate_point_garch_combinations(
    *,
    point_predictions: dict[str, pd.DataFrame],
    garch_results: pd.DataFrame,
    garch_predictions: pd.DataFrame,
    garch_model_name: str | None = None,
    garch_model_names: list[str] | None = None,
    garch_selection: str = "all",
    garch_metric: str = "test_nll",
) -> pd.DataFrame:
    requested_models = garch_model_names or ([garch_model_name] if garch_model_name else None)
    selected_garch_models = select_garch_models(
        garch_results,
        requested_models=requested_models,
        metric=garch_metric,
        selection=garch_selection,
    )

    rows = []
    for selected_garch in selected_garch_models:
        garch_meta = garch_results.loc[garch_results["model_name"].astype(str).eq(selected_garch)]
        if garch_meta.empty:
            raise ValueError(f"GARCH model not found in results: {selected_garch}")
        nu = float(garch_meta.iloc[0].get("nu", np.nan))
        selected_garch_predictions = garch_predictions.loc[
            garch_predictions["model_name"].astype(str).eq(selected_garch)
        ].copy()
        if selected_garch_predictions.empty:
            raise ValueError(f"GARCH predictions not found for model: {selected_garch}")

        for point_name, point_frame in point_predictions.items():
            merged = align_point_and_garch(point_frame, selected_garch_predictions)
            y_true = pd.to_numeric(merged["y_true"], errors="coerce").to_numpy(dtype=np.float64)
            mean = pd.to_numeric(merged["y_pred"], errors="coerce").to_numpy(dtype=np.float64)
            garch_sigma = pd.to_numeric(merged["volatility_forecast"], errors="coerce").to_numpy(dtype=np.float64)

            rows.append(
                metric_row_from_distribution(
                    model_name=f"{point_name}+{selected_garch}_studentst",
                    point_model=point_name,
                    garch_model=selected_garch,
                    distribution="studentst",
                    y_true=y_true,
                    mean=mean,
                    sigma=garch_sigma,
                    quantiles=student_t_quantiles(mean, garch_sigma, nu),
                    nu=nu,
                )
            )

    for point_name, point_frame in point_predictions.items():
        y_true = pd.to_numeric(point_frame["y_true"], errors="coerce").to_numpy(dtype=np.float64)
        mean = pd.to_numeric(point_frame["y_pred"], errors="coerce").to_numpy(dtype=np.float64)

        if "sigma" in point_frame.columns:
            point_sigma = pd.to_numeric(point_frame["sigma"], errors="coerce").to_numpy(dtype=np.float64)
            rows.append(
                metric_row_from_distribution(
                    model_name=f"{point_name}_native_normal",
                    point_model=point_name,
                    garch_model=None,
                    distribution="normal",
                    y_true=y_true,
                    mean=mean,
                    sigma=point_sigma,
                    quantiles=normal_quantiles(mean, point_sigma),
                )
            )

        if {"q05", "q25", "q50", "q75", "q95"}.issubset(point_frame.columns):
            native_quantiles = {
                alpha: pd.to_numeric(point_frame[f"q{int(alpha * 100):02d}"], errors="coerce").to_numpy(dtype=np.float64)
                for alpha in QUANTILE_ALPHAS
            }
            rows.append(
                metric_row_from_distribution(
                    model_name=f"{point_name}_native_quantiles",
                    point_model=point_name,
                    garch_model=None,
                    distribution="normal",
                    y_true=y_true,
                    mean=pd.to_numeric(point_frame["q50"], errors="coerce").to_numpy(dtype=np.float64),
                    sigma=np.maximum((native_quantiles[0.95] - native_quantiles[0.05]) / (2.0 * norm.ppf(0.95)), 1e-12),
                    quantiles=native_quantiles,
                )
            )

    return pd.DataFrame(rows).sort_values(["NLL", "CRPS_approx", "model_name"], na_position="last")


def load_point_predictions_from_s3(
    *,
    s3: Any,
    bucket: str,
    stage1_prefix: str,
    point_results: pd.DataFrame,
    top_n: int,
    point_metric: str,
) -> dict[str, pd.DataFrame]:
    selected = select_top_point_models(point_results, top_n=top_n, metric=point_metric)
    predictions = {}
    for row in selected.to_dict("records"):
        job_id = str(row["job_id"])
        key = str(row.get("predictions_key") or f"{stage1_prefix}/jobs/{job_id}/predictions.parquet")
        predictions[job_id] = read_parquet(s3, bucket, key)
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate probabilistic forecasts formed by point forecasts plus GARCH volatility."
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--dataset-prefix", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--suite", default="fast")
    parser.add_argument("--results-prefix", default=DEFAULT_RESULTS_PREFIX)
    parser.add_argument("--stage1-run-id", default="latest")
    parser.add_argument("--garch-run-id", default="latest")
    parser.add_argument("--garch-subdir", default=DEFAULT_GARCH_SUBDIR)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--top-n-point", type=int, default=5)
    parser.add_argument("--point-metric", default="RMSE")
    parser.add_argument(
        "--garch-selection",
        default="all",
        choices=["all", "best"],
        help="Use every GARCH model from garch_results, or only the best by --garch-metric.",
    )
    parser.add_argument("--garch-metric", default="test_nll")
    parser.add_argument(
        "--garch-model",
        action="append",
        default=None,
        help="Specific GARCH model to use. Can be repeated or passed as a comma-separated list.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    s3 = make_s3_client()
    stage1_prefix = resolve_stage1_prefix(
        results_prefix=args.results_prefix,
        suite=args.suite,
        horizon=args.horizon,
        stage1_run_id=args.stage1_run_id,
    )
    garch_prefix = resolve_garch_prefix(
        dataset_prefix=args.dataset_prefix,
        garch_subdir=args.garch_subdir,
        garch_run_id=args.garch_run_id,
    )

    point_results = read_parquet(s3, args.bucket, f"{stage1_prefix}/stage1_results.parquet")
    garch_results = read_parquet(s3, args.bucket, f"{garch_prefix}/garch_results.parquet")
    garch_predictions = read_parquet(s3, args.bucket, f"{garch_prefix}/garch_predictions.parquet")
    point_predictions = load_point_predictions_from_s3(
        s3=s3,
        bucket=args.bucket,
        stage1_prefix=stage1_prefix,
        point_results=point_results,
        top_n=args.top_n_point,
        point_metric=args.point_metric,
    )

    results = evaluate_point_garch_combinations(
        point_predictions=point_predictions,
        garch_results=garch_results,
        garch_predictions=garch_predictions,
        garch_model_names=parse_requested_models(args.garch_model),
        garch_selection=args.garch_selection,
        garch_metric=args.garch_metric,
    )
    forecasts = build_point_garch_forecasts(
        point_predictions=point_predictions,
        garch_results=garch_results,
        garch_predictions=garch_predictions,
        garch_model_names=parse_requested_models(args.garch_model),
        garch_selection=args.garch_selection,
        garch_metric=args.garch_metric,
    )
    print(results.to_string(index=False))
    if args.dry_run:
        return

    output_prefix = (
        f"{args.dataset_prefix.strip('/')}/{args.output_subdir.strip('/')}/"
        f"horizon_{args.horizon}/{args.stage1_run_id}_garch_{args.garch_run_id}"
    )
    upload_parquet(s3, args.bucket, f"{output_prefix}/probabilistic_combination_results.parquet", results)
    upload_csv(s3, args.bucket, f"{output_prefix}/probabilistic_combination_results.csv", results)
    upload_parquet(s3, args.bucket, f"{output_prefix}/probabilistic_combination_predictions.parquet", forecasts)
    upload_bytes(
        s3,
        args.bucket,
        f"{output_prefix}/run_config.json",
        json.dumps(vars(args), ensure_ascii=False, indent=2).encode("utf-8"),
        "application/json",
    )
    print(f"uploaded=s3://{args.bucket}/{output_prefix}/")


if __name__ == "__main__":
    main()
