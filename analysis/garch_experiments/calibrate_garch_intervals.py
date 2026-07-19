from __future__ import annotations

import argparse
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t


DEFAULT_BUCKET = "binance-data-downloader"
DEFAULT_GARCH_SUBDIR = "garch_experiments"
DEFAULT_OUTPUT_PREFIX = "garch_interval_calibration"
DEFAULT_COVERAGE = 0.90
QUANTILE_ALPHAS = (0.05, 0.25, 0.50, 0.75, 0.95)
TARGET_SPECS = {
    10: ("dataset_target_10", "target_log_return_10m"),
    20: ("dataset_target_20/with_price_hmm_n4", "target_log_return_20m"),
    30: ("dataset_target_30", "target_log_return_30m"),
}


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


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


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


def garch_prefix(dataset_prefix: str, garch_subdir: str, run_id: str) -> str:
    return f"{dataset_prefix.strip('/')}/{garch_subdir.strip('/')}/{run_id.strip('/')}"


def output_prefix(root: str, horizon: int, run_id: str) -> str:
    return f"{root.strip('/')}/horizon_{horizon}/{run_id.strip('/')}"


def latest_prefix(root: str, horizon: int) -> str:
    return f"{root.strip('/')}/horizon_{horizon}/latest"


def student_t_quantile_multiplier(alpha: float, nu: float) -> float:
    if not np.isfinite(nu) or nu <= 2.0:
        return np.nan
    return math.sqrt((nu - 2.0) / nu) * float(t.ppf(alpha, df=nu))


def student_t_quantiles(mean: np.ndarray, sigma: np.ndarray, nu: float) -> dict[float, np.ndarray]:
    return {
        alpha: mean + sigma * student_t_quantile_multiplier(alpha, nu)
        for alpha in QUANTILE_ALPHAS
    }


def pinball_loss(y_true: np.ndarray, y_quantile: np.ndarray, alpha: float) -> float:
    delta = y_true - y_quantile
    return float(np.mean(np.maximum(alpha * delta, (alpha - 1.0) * delta)))


def quantile_crps_approx(y_true: np.ndarray, quantiles: dict[float, np.ndarray]) -> float:
    losses = np.array([2.0 * pinball_loss(y_true, quantiles[alpha], alpha) for alpha in QUANTILE_ALPHAS])
    return float(np.trapezoid(losses, np.array(QUANTILE_ALPHAS, dtype=np.float64)))


def split_validation_eval(frame: pd.DataFrame, validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(frame)
    if n_rows < 2:
        raise ValueError("Need at least two rows for validation/eval split")
    split = int(n_rows * validation_fraction)
    split = min(max(split, 1), n_rows - 1)
    return np.arange(split), np.arange(split, n_rows)


def coverage_width(
    y_true: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    *,
    nu: float,
    sigma_scale: float,
    coverage: float,
) -> tuple[float, float]:
    lower_alpha = (1.0 - coverage) / 2.0
    upper_alpha = 1.0 - lower_alpha
    scaled_sigma = np.maximum(np.asarray(sigma, dtype=np.float64) * sigma_scale, 1e-12)
    lower = mean + scaled_sigma * student_t_quantile_multiplier(lower_alpha, nu)
    upper = mean + scaled_sigma * student_t_quantile_multiplier(upper_alpha, nu)
    return (
        float(((y_true >= lower) & (y_true <= upper)).mean()),
        float(np.mean(upper - lower)),
    )


def valid_garch_row(row: pd.Series, args: argparse.Namespace, predictions: pd.DataFrame) -> tuple[bool, str]:
    if bool(row.get("converged")) is not True:
        return False, "not_converged"
    nu = float(row.get("nu", np.nan))
    if not np.isfinite(nu) or nu <= 2.0:
        return False, "invalid_nu"
    std_resid_std = float(row.get("std_resid_std", np.nan))
    if not np.isfinite(std_resid_std) or not (args.min_std_resid_std <= std_resid_std <= args.max_std_resid_std):
        return False, "std_resid_std_out_of_bounds"
    y_std = float(np.nanstd(predictions["y_true"].to_numpy(dtype=np.float64)))
    sigma_mean = float(np.nanmean(predictions["volatility_forecast"].to_numpy(dtype=np.float64)))
    if not np.isfinite(sigma_mean) or sigma_mean <= 0.0:
        return False, "invalid_sigma_mean"
    if y_std > 0 and sigma_mean / y_std > args.max_sigma_mean_to_y_std:
        return False, "sigma_mean_too_large"
    mean_abs = float(np.nanmean(np.abs(predictions["mean_forecast"].to_numpy(dtype=np.float64))))
    if y_std > 0 and mean_abs / y_std > args.max_abs_mean_forecast_to_y_std:
        return False, "mean_forecast_too_large"
    return True, "ok"


def select_scale(
    y_true: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    *,
    nu: float,
    coverage: float,
    scale_grid: np.ndarray,
) -> dict[str, float]:
    rows = []
    for scale in scale_grid:
        actual_coverage, width = coverage_width(
            y_true,
            mean,
            sigma,
            nu=nu,
            sigma_scale=float(scale),
            coverage=coverage,
        )
        rows.append(
            {
                "sigma_scale": float(scale),
                "coverage": actual_coverage,
                "coverage_error_abs": abs(actual_coverage - coverage),
                "interval_width": width,
            }
        )
    ranked = pd.DataFrame(rows).sort_values(
        ["coverage_error_abs", "interval_width", "sigma_scale"],
        ascending=[True, True, True],
    )
    return ranked.iloc[0].to_dict()


def metric_row(
    *,
    horizon: int,
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    nu: float,
    sigma_scale: float,
    coverage: float,
) -> dict[str, Any]:
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64) * sigma_scale, 1e-12)
    mean = np.asarray(mean, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    residual = y_true - mean
    quantiles = student_t_quantiles(mean, sigma, nu)
    lower_alpha = (1.0 - coverage) / 2.0
    upper_alpha = 1.0 - lower_alpha
    lower = mean + sigma * student_t_quantile_multiplier(lower_alpha, nu)
    upper = mean + sigma * student_t_quantile_multiplier(upper_alpha, nu)
    logpdf = standardized_student_logpdf(residual / sigma, nu) - np.log(sigma)
    return {
        "horizon": horizon,
        "model_name": model_name,
        "split": split_name,
        "n_rows": int(len(y_true)),
        "nu": float(nu),
        "sigma_scale": float(sigma_scale),
        "mean_error": float(np.mean(residual)),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(math.sqrt(np.mean(np.square(residual)))),
        "NLL": float(-np.nanmean(logpdf)),
        "CRPS_approx": quantile_crps_approx(y_true, quantiles),
        "PinballLoss_05": pinball_loss(y_true, quantiles[0.05], 0.05),
        "PinballLoss_50": pinball_loss(y_true, quantiles[0.50], 0.50),
        "PinballLoss_95": pinball_loss(y_true, quantiles[0.95], 0.95),
        "Coverage90": float(((y_true >= lower) & (y_true <= upper)).mean()),
        "IntervalWidth90": float(np.mean(upper - lower)),
        "sigma_mean": float(np.mean(sigma)),
        "std_resid_std": float(np.nanstd(residual / sigma)),
    }


def calibrated_predictions(
    *,
    horizon: int,
    model_name: str,
    predictions: pd.DataFrame,
    nu: float,
    sigma_scale: float,
) -> pd.DataFrame:
    frame = predictions.copy()
    mean = frame["mean_forecast"].to_numpy(dtype=np.float64)
    sigma = np.maximum(frame["volatility_forecast"].to_numpy(dtype=np.float64) * sigma_scale, 1e-12)
    quantiles = student_t_quantiles(mean, sigma, nu)
    frame.insert(0, "horizon", horizon)
    frame["model_name"] = model_name
    frame["nu"] = float(nu)
    frame["sigma_scale"] = float(sigma_scale)
    frame["volatility_forecast_raw"] = frame["volatility_forecast"]
    frame["volatility_forecast"] = sigma
    frame["variance_forecast"] = np.square(sigma)
    for alpha, values in quantiles.items():
        frame[f"q{int(alpha * 100):02d}"] = values
    return frame


def calibrate_horizon(
    *,
    s3: Any,
    bucket: str,
    horizon: int,
    dataset_prefix: str,
    garch_subdir: str,
    garch_run_id: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefix = garch_prefix(dataset_prefix, garch_subdir, garch_run_id)
    results = read_parquet(s3, bucket, f"{prefix}/garch_results.parquet")
    predictions = read_parquet(s3, bucket, f"{prefix}/garch_predictions.parquet")
    run_config = read_json(s3, bucket, f"{prefix}/run_config.json")
    scale_grid = np.arange(args.scale_min, args.scale_max + args.scale_step / 2.0, args.scale_step)

    metric_rows = []
    scale_rows = []
    prediction_frames = []
    for _, row in results.iterrows():
        model_name = str(row["model_name"])
        model_predictions = predictions.loc[predictions["model_name"].astype(str).eq(model_name)].copy()
        if model_predictions.empty:
            continue
        is_valid, reason = valid_garch_row(row, args, model_predictions)
        if not is_valid:
            scale_rows.append(
                {
                    "horizon": horizon,
                    "model_name": model_name,
                    "status": "excluded",
                    "exclude_reason": reason,
                    "sigma_scale": np.nan,
                }
            )
            continue

        validation_idx, eval_idx = split_validation_eval(model_predictions, args.validation_fraction)
        y = model_predictions["y_true"].to_numpy(dtype=np.float64)
        mean = model_predictions["mean_forecast"].to_numpy(dtype=np.float64)
        sigma = model_predictions["volatility_forecast"].to_numpy(dtype=np.float64)
        nu = float(row["nu"])
        selected = select_scale(
            y[validation_idx],
            mean[validation_idx],
            sigma[validation_idx],
            nu=nu,
            coverage=args.coverage,
            scale_grid=scale_grid,
        )
        sigma_scale = float(selected["sigma_scale"])
        scale_rows.append(
            {
                "horizon": horizon,
                "model_name": model_name,
                "status": "included",
                "exclude_reason": "",
                "sigma_scale": sigma_scale,
                "validation_coverage": selected["coverage"],
                "validation_interval_width": selected["interval_width"],
                "target_coverage": args.coverage,
                "raw_std_resid_std": float(row.get("std_resid_std", np.nan)),
                "raw_test_nll": float(row.get("test_nll", np.nan)),
                "garch_run_id": run_config.get("run_id", garch_run_id),
            }
        )
        metric_rows.append(
            metric_row(
                horizon=horizon,
                model_name=model_name,
                split_name="validation",
                y_true=y[validation_idx],
                mean=mean[validation_idx],
                sigma=sigma[validation_idx],
                nu=nu,
                sigma_scale=sigma_scale,
                coverage=args.coverage,
            )
        )
        metric_rows.append(
            metric_row(
                horizon=horizon,
                model_name=model_name,
                split_name="eval",
                y_true=y[eval_idx],
                mean=mean[eval_idx],
                sigma=sigma[eval_idx],
                nu=nu,
                sigma_scale=sigma_scale,
                coverage=args.coverage,
            )
        )
        prediction_frames.append(
            calibrated_predictions(
                horizon=horizon,
                model_name=model_name,
                predictions=model_predictions,
                nu=nu,
                sigma_scale=sigma_scale,
            )
        )

    metrics = pd.DataFrame(metric_rows)
    scales = pd.DataFrame(scale_rows)
    calibrated = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return metrics, scales, calibrated


def save_outputs(
    *,
    s3: Any,
    bucket: str,
    root: str,
    horizon: int,
    run_id: str,
    metrics: pd.DataFrame,
    scales: pd.DataFrame,
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    prefixes = [output_prefix(root, horizon, run_id), latest_prefix(root, horizon)]
    for prefix in prefixes:
        upload_parquet(s3, bucket, f"{prefix}/calibrated_garch_metrics.parquet", metrics)
        upload_csv(s3, bucket, f"{prefix}/calibrated_garch_metrics.csv", metrics)
        upload_parquet(s3, bucket, f"{prefix}/garch_sigma_scales.parquet", scales)
        upload_csv(s3, bucket, f"{prefix}/garch_sigma_scales.csv", scales)
        upload_parquet(s3, bucket, f"{prefix}/calibrated_garch_predictions.parquet", predictions)
        upload_bytes(
            s3,
            bucket,
            f"{prefix}/run_config.json",
            json.dumps(config, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
            "application/json",
        )
    return f"s3://{bucket}/{output_prefix(root, horizon, run_id)}/"


def horizon_arg(value: str) -> int | str:
    if value == "all":
        return value
    horizon = int(value)
    if horizon not in TARGET_SPECS:
        allowed = ", ".join(str(item) for item in sorted(TARGET_SPECS))
        raise argparse.ArgumentTypeError(f"horizon must be one of: {allowed}, all")
    return horizon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate GARCH Student-t interval widths by sigma scaling.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--horizon", type=horizon_arg, default="all")
    parser.add_argument("--dataset-prefix", default=None)
    parser.add_argument("--garch-subdir", default=DEFAULT_GARCH_SUBDIR)
    parser.add_argument("--garch-run-id", default="latest")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.50)
    parser.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    parser.add_argument("--scale-min", type=float, default=0.50)
    parser.add_argument("--scale-max", type=float, default=1.25)
    parser.add_argument("--scale-step", type=float, default=0.01)
    parser.add_argument("--min-std-resid-std", type=float, default=0.50)
    parser.add_argument("--max-std-resid-std", type=float, default=1.50)
    parser.add_argument("--max-sigma-mean-to-y-std", type=float, default=5.0)
    parser.add_argument("--max-abs-mean-forecast-to-y-std", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or make_run_id()
    s3 = make_s3_client()
    horizons = sorted(TARGET_SPECS) if args.horizon == "all" else [int(args.horizon)]
    if args.dataset_prefix and len(horizons) > 1:
        raise ValueError("--dataset-prefix can only be used with a single horizon")

    all_metrics = []
    all_scales = []
    locations = []
    for horizon in horizons:
        dataset_prefix = args.dataset_prefix or TARGET_SPECS[horizon][0]
        metrics, scales, predictions = calibrate_horizon(
            s3=s3,
            bucket=args.bucket,
            horizon=horizon,
            dataset_prefix=dataset_prefix,
            garch_subdir=args.garch_subdir,
            garch_run_id=args.garch_run_id,
            args=args,
        )
        all_metrics.append(metrics)
        all_scales.append(scales)
        print(f"\nHORIZON {horizon}")
        if not scales.empty:
            print(scales.to_string(index=False))
        if not metrics.empty:
            print(metrics.sort_values(["split", "Coverage90", "IntervalWidth90"]).to_string(index=False))
        if not args.dry_run:
            config = {
                **vars(args),
                "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_prefix": dataset_prefix,
                "horizon": horizon,
            }
            locations.append(
                save_outputs(
                    s3=s3,
                    bucket=args.bucket,
                    root=args.output_prefix,
                    horizon=horizon,
                    run_id=run_id,
                    metrics=metrics,
                    scales=scales,
                    predictions=predictions,
                    config=config,
                )
            )

    if all_metrics:
        summary = pd.concat(all_metrics, ignore_index=True)
        print("\nSUMMARY")
        print(summary.sort_values(["horizon", "split", "NLL", "CRPS_approx"]).to_string(index=False))
    if locations:
        print("\nUPLOADED")
        for location in locations:
            print(location)


if __name__ == "__main__":
    main()
