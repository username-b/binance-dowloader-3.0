from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from arch import arch_model
from catboost import CatBoostRegressor
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits


DEFAULT_BUCKET = "binance-data-downloader"
DEFAULT_OUTPUT_PREFIX = "trading_strategy_experiment"
DEFAULT_RESULTS_SUBDIR = "model_experiments"
DEFAULT_TIMESTAMP_COLUMN = "timestamp"
DEFAULT_FEE_PER_ACTION = 0.0005
DEFAULT_VALIDATION_FRACTION = 0.5
DEFAULT_PROBABILITY_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)
QUANTILE_ALPHAS = (0.05, 0.25, 0.50, 0.75, 0.95)


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "build_price_feature_day.py").exists():
            return candidate
    raise FileNotFoundError("Could not find project root")


PROJECT_ROOT = find_project_root()
for module_dir in (
    PROJECT_ROOT,
    PROJECT_ROOT / "analysis" / "target_10_model_experiments",
    PROJECT_ROOT / "analysis" / "stage1_model_search",
):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from build_price_feature_day import load_s3_parquet, make_s3_client  # noqa: E402
from evaluate_stage3_hmm_lift import (  # noqa: E402
    STAGE3_NAME,
    TARGET_SPECS,
    TargetedModelSpec,
    model_family_by_name,
    select_stage3_model,
)
from train_stage1_architecture_search import (  # noqa: E402
    PROBABILISTIC_MODEL_CONFIGS,
    prepare_data as prepare_ml_data,
)
from train_target_10_models import build_estimator, prepare_target  # noqa: E402


@dataclass(frozen=True)
class ExperimentSpec:
    horizon: int
    criterion: str
    family_label: str
    n_features: int
    garch_name: str


EXPERIMENT_SPECS = {
    10: ExperimentSpec(10, "MAE", "lasso", 12, "gjr_garch_1_1"),
    20: ExperimentSpec(20, "DA", "huber", 33, "gjr_garch_1_1"),
    30: ExperimentSpec(30, "DA", "huber", 23, "garch_1_1"),
}

ML_DATASET_PREFIXES = {
    10: "dataset_target_10",
    20: "dataset_target_20/with_price_hmm_n4",
    30: "dataset_target_30",
}


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("trading_strategy_experiment")


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def upload_bytes(s3: Any, bucket: str, key: str, payload: bytes, content_type: str) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)


def upload_json(s3: Any, bucket: str, key: str, payload: dict[str, Any]) -> None:
    upload_bytes(
        s3,
        bucket,
        key,
        json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        "application/json",
    )


def upload_parquet(s3: Any, bucket: str, key: str, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, engine="pyarrow", compression="zstd")
    upload_bytes(s3, bucket, key, buffer.getvalue(), "application/vnd.apache.parquet")


def upload_csv(s3: Any, bucket: str, key: str, frame: pd.DataFrame) -> None:
    upload_bytes(s3, bucket, key, frame.to_csv(index=False).encode("utf-8-sig"), "text/csv; charset=utf-8")


def serialize_joblib(value: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(value, buffer, compress=3)
    return buffer.getvalue()


def read_required_parquet(s3: Any, bucket: str, key: str) -> pd.DataFrame:
    frame = load_s3_parquet(s3, bucket, key)
    if frame is None:
        raise FileNotFoundError(f"s3://{bucket}/{key}")
    return frame


def load_train_test(s3: Any, bucket: str, dataset_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        read_required_parquet(s3, bucket, f"{dataset_prefix}/train.parquet"),
        read_required_parquet(s3, bucket, f"{dataset_prefix}/test.parquet"),
    )


def fit_selected_point_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    baseline_results: pd.DataFrame,
    spec: ExperimentSpec,
    target_column: str,
) -> tuple[Any, pd.Series, np.ndarray, pd.DataFrame, dict[str, Any]]:
    selection_spec = TargetedModelSpec(spec.horizon, spec.criterion, spec.family_label, spec.n_features)
    selected = select_stage3_model(baseline_results, selection_spec)
    features = json.loads(selected["features"])
    y_train = prepare_target(train, target_column)
    y_test = prepare_target(test, target_column)
    train_mask, test_mask = y_train.notna(), y_test.notna()
    train_X = train.loc[train_mask, features]
    train_y = y_train.loc[train_mask].to_numpy(dtype=float)
    if len(train_y) <= spec.horizon:
        raise ValueError("Training split is too short for the horizon purge gap")
    fit_size = len(train_y) - spec.horizon
    estimator = build_estimator(model_family_by_name(str(selected["model_family"])), len(features))
    estimator.fit(train_X.iloc[:fit_size], train_y[:fit_size])
    train_pred = estimator.predict(train_X)
    test_pred = estimator.predict(test.loc[test_mask, features])
    predictions = pd.DataFrame(
        {
            "row_id": np.arange(int(test_mask.sum()), dtype=np.int64),
            "timestamp": test.loc[test_mask, DEFAULT_TIMESTAMP_COLUMN].to_numpy(),
            "y_true": y_test.loc[test_mask].to_numpy(dtype=float),
            "point_mean": np.asarray(test_pred, dtype=float),
        }
    )
    train_residual = train_y - train_pred
    garch_fit_residual = pd.Series(train_residual[:fit_size])
    delayed_residual_prefix = np.asarray(train_residual[fit_size:], dtype=float)
    metadata = {
        "selected_model_id": selected["model_id"],
        "model_family": selected["model_family"],
        "criterion": spec.criterion,
        "features": features,
        "n_features": len(features),
        "purge_gap_rows": spec.horizon,
        "fit_rows": fit_size,
    }
    return estimator, garch_fit_residual, delayed_residual_prefix, predictions, metadata


def fit_residual_garch(
    fit_residual: pd.Series,
    delayed_residual_prefix: np.ndarray,
    test_residual: np.ndarray,
    model_name: str,
    scale: float,
    maxiter: int,
) -> tuple[Any, np.ndarray, float, dict[str, Any]]:
    if model_name == "gjr_garch_1_1":
        vol, p, o, q = "GARCH", 1, 1, 1
    elif model_name == "garch_1_1":
        vol, p, o, q = "GARCH", 1, 0, 1
    else:
        raise ValueError(f"Unsupported GARCH model: {model_name}")
    train_values = fit_residual.to_numpy(dtype=float) * scale
    test_values = np.asarray(test_residual, dtype=float) * scale
    model = arch_model(train_values, mean="Zero", vol=vol, p=p, o=o, q=q, dist="studentst", rescale=False)
    result = model.fit(disp="off", options={"maxiter": maxiter})
    params = result.params
    omega = float(params.get("omega", 0.0))
    alpha = float(params.get("alpha[1]", 0.0))
    gamma = float(params.get("gamma[1]", 0.0))
    beta = float(params.get("beta[1]", 0.0))
    previous_variance = float(np.square(np.asarray(result.conditional_volatility, dtype=float)[-1]))
    observable_residuals = np.concatenate(
        [np.asarray(delayed_residual_prefix, dtype=float) * scale, test_values]
    )
    delay = len(delayed_residual_prefix)
    variance_forecasts = np.empty(len(test_values), dtype=float)
    for index in range(len(test_values)):
        # At test row i only the outcome whose origin is i-h is observable.
        shock = float(observable_residuals[index])
        shock_square = shock * shock
        next_variance = omega + alpha * shock_square + beta * previous_variance
        if o:
            next_variance += gamma * shock_square * float(shock < 0.0)
        previous_variance = max(float(next_variance), 1e-18)
        variance_forecasts[index] = previous_variance
    sigma = np.sqrt(variance_forecasts) / scale
    nu = float(result.params.get("nu", np.nan))
    metadata = {
        "model_name": model_name,
        "vol": vol,
        "p": p,
        "o": o,
        "q": q,
        "distribution": "studentst",
        "nu": nu,
        "scale": scale,
        "causal_update_delay_rows": delay,
        "forecast_method": "manual delayed residual recursion",
        "convergence_flag": int(getattr(result, "convergence_flag", -1)),
        "params": {str(k): float(v) for k, v in result.params.items()},
    }
    return result, sigma, nu, metadata


def standardized_student_cdf(x: np.ndarray, mean: np.ndarray, sigma: np.ndarray, nu: float) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    standardizer = math.sqrt((nu - 2.0) / nu)
    return student_t.cdf((np.asarray(x) - np.asarray(mean)) / (sigma * standardizer), df=nu)


def fit_catboost_multiquantile(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    horizon: int,
    threads: int,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    require_hmm = horizon == 20 and any(c.startswith("price_hmm_n4_") for c in train.columns)
    data = prepare_ml_data(
        train,
        test,
        target_column,
        hmm_feature_prefix="price_hmm_n4",
        require_hmm_features=require_hmm,
    )
    params = {
        **PROBABILISTIC_MODEL_CONFIGS["catboost_quantile"],
        "loss_function": "MultiQuantile:alpha=" + ",".join(str(a) for a in QUANTILE_ALPHAS),
        "random_seed": 42,
        "thread_count": threads,
        "verbose": False,
    }
    if len(data.y_train) <= horizon:
        raise ValueError("Training split is too short for the horizon purge gap")
    fit_size = len(data.y_train) - horizon
    model = CatBoostRegressor(**params)
    model.fit(data.X_train.iloc[:fit_size], data.y_train[:fit_size], verbose=False)
    values = np.asarray(model.predict(data.X_test), dtype=float)
    if values.ndim != 2 or values.shape[1] != len(QUANTILE_ALPHAS):
        raise RuntimeError(f"Unexpected CatBoost prediction shape: {values.shape}")
    frame = pd.DataFrame({f"q{int(a * 100):02d}": values[:, i] for i, a in enumerate(QUANTILE_ALPHAS)})
    frame.insert(0, "timestamp", data.test_timestamps)
    metadata = {
        "params": params,
        "features": data.feature_columns,
        "n_features": len(data.feature_columns),
        "purge_gap_rows": horizon,
        "fit_rows": fit_size,
    }
    return model, frame, metadata


def quantile_cdf_at(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    q = frame[[f"q{int(a * 100):02d}" for a in QUANTILE_ALPHAS]].to_numpy(dtype=float)
    q = np.maximum.accumulate(q, axis=1)
    result = np.empty(len(q), dtype=float)
    for i, row in enumerate(q):
        if threshold <= row[0]:
            result[i] = QUANTILE_ALPHAS[0]
        elif threshold >= row[-1]:
            result[i] = QUANTILE_ALPHAS[-1]
        else:
            j = int(np.searchsorted(row, threshold, side="right") - 1)
            width = row[j + 1] - row[j]
            weight = 0.5 if width <= 1e-15 else (threshold - row[j]) / width
            result[i] = QUANTILE_ALPHAS[j] + weight * (QUANTILE_ALPHAS[j + 1] - QUANTILE_ALPHAS[j])
    return result


def make_signal(long_probability: np.ndarray, short_probability: np.ndarray, probability_threshold: float) -> np.ndarray:
    long_probability = np.asarray(long_probability, dtype=float)
    short_probability = np.asarray(short_probability, dtype=float)
    signal = np.zeros(len(long_probability), dtype=np.int8)
    long_mask = (long_probability >= probability_threshold) & (long_probability >= short_probability)
    short_mask = (short_probability >= probability_threshold) & (short_probability > long_probability)
    signal[long_mask] = 1
    signal[short_mask] = -1
    return signal


def split_segments(frame: pd.DataFrame, validation_fraction: float) -> pd.Series:
    split = max(1, min(len(frame) - 1, int(len(frame) * validation_fraction)))
    return pd.Series(np.where(np.arange(len(frame)) < split, "validation", "evaluation"), index=frame.index)


def simulate_non_overlapping(
    predictions: pd.DataFrame,
    signal: np.ndarray,
    *,
    horizon: int,
    fee_per_action: float,
    strategy: str,
    segment: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = predictions.loc[predictions["segment"].eq(segment)].copy().reset_index(drop=True)
    local_signal = np.asarray(signal)[predictions["segment"].eq(segment).to_numpy()]
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    holding_period = pd.Timedelta(minutes=int(horizon))
    next_allowed = None
    trades = []
    round_trip_fee = 2.0 * fee_per_action
    for i, side in enumerate(local_signal):
        if side == 0:
            continue
        timestamp = timestamps.iloc[i]
        if next_allowed is not None and timestamp < next_allowed:
            continue
        underlying_return = math.expm1(float(frame.loc[i, "y_true"]))
        gross_return = float(side * underlying_return)
        net_return = gross_return - round_trip_fee
        trades.append(
            {
                "strategy": strategy,
                "segment": segment,
                "horizon": horizon,
                "entry_timestamp": timestamp,
                "exit_timestamp": timestamp + holding_period,
                "side": int(side),
                "gross_return": gross_return,
                "commission": round_trip_fee,
                "net_return": net_return,
            }
        )
        next_allowed = timestamp + holding_period
    trades_frame = pd.DataFrame(trades)
    if trades_frame.empty:
        metrics = empty_metrics(strategy, segment, horizon)
        return trades_frame, pd.DataFrame(), metrics
    multipliers = np.maximum(1.0 + trades_frame["net_return"].to_numpy(dtype=float), 1e-12)
    equity = np.cumprod(multipliers)
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / peaks - 1.0
    equity_frame = trades_frame[["exit_timestamp"]].copy()
    equity_frame["strategy"] = strategy
    equity_frame["segment"] = segment
    equity_frame["equity"] = equity
    equity_frame["drawdown"] = drawdown
    net = trades_frame["net_return"].to_numpy(dtype=float)
    positive = net[net > 0].sum()
    negative = -net[net < 0].sum()
    metrics = {
        "strategy": strategy,
        "segment": segment,
        "horizon": horizon,
        "trades": len(trades_frame),
        "long_trades": int((trades_frame["side"] == 1).sum()),
        "short_trades": int((trades_frame["side"] == -1).sum()),
        "gross_total_return": float(np.prod(1.0 + trades_frame["gross_return"]) - 1.0),
        "net_total_return": float(equity[-1] - 1.0),
        "mean_net_return": float(np.mean(net)),
        "median_net_return": float(np.median(net)),
        "win_rate": float(np.mean(net > 0)),
        "profit_factor": float(positive / negative) if negative > 0 else np.inf,
        "max_drawdown": float(drawdown.min()),
        "total_commission": float(trades_frame["commission"].sum()),
    }
    return trades_frame, equity_frame, metrics


def empty_metrics(strategy: str, segment: str, horizon: int) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "segment": segment,
        "horizon": horizon,
        "trades": 0,
        "long_trades": 0,
        "short_trades": 0,
        "gross_total_return": 0.0,
        "net_total_return": 0.0,
        "mean_net_return": np.nan,
        "median_net_return": np.nan,
        "win_rate": np.nan,
        "profit_factor": np.nan,
        "max_drawdown": 0.0,
        "total_commission": 0.0,
    }


def select_probability_threshold(metrics: pd.DataFrame, model_prefix: str, minimum_trades: int) -> float:
    candidates = metrics.loc[
        metrics["segment"].eq("validation")
        & metrics["strategy"].str.startswith(model_prefix)
        & metrics["trades"].ge(minimum_trades)
    ].copy()
    if candidates.empty:
        candidates = metrics.loc[
            metrics["segment"].eq("validation") & metrics["strategy"].str.startswith(model_prefix)
        ].copy()
    if candidates.empty:
        return np.nan
    candidates["probability_threshold"] = candidates["strategy"].str.rsplit("_p", n=1).str[-1].astype(float)
    return float(candidates.sort_values(["net_total_return", "probability_threshold"], ascending=[False, False]).iloc[0]["probability_threshold"])


def run_backtests(
    predictions: pd.DataFrame,
    *,
    horizon: int,
    fee_per_action: float,
    probability_thresholds: Iterable[float],
    validation_fraction: float,
    minimum_validation_trades: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = predictions.sort_values("timestamp").reset_index(drop=True).copy()
    frame["segment"] = split_segments(frame, validation_fraction)
    cost = 2.0 * fee_per_action
    strategies: dict[str, np.ndarray] = {
        "point_sign": np.sign(frame["point_mean"].to_numpy(dtype=float)).astype(np.int8),
        "point_cost_threshold": np.where(
            frame["point_mean"] > cost, 1, np.where(frame["point_mean"] < -cost, -1, 0)
        ).astype(np.int8),
    }
    for threshold in probability_thresholds:
        strategies[f"stat_garch_p{threshold:.2f}"] = make_signal(
            frame["garch_p_long"], frame["garch_p_short"], threshold
        )
        strategies[f"catboost_mq_p{threshold:.2f}"] = make_signal(
            frame["catboost_p_long"], frame["catboost_p_short"], threshold
        )
    trade_frames, equity_frames, metric_rows = [], [], []
    for strategy, signal in strategies.items():
        for segment in ("validation", "evaluation"):
            trades, equity, metrics = simulate_non_overlapping(
                frame,
                signal,
                horizon=horizon,
                fee_per_action=fee_per_action,
                strategy=strategy,
                segment=segment,
            )
            trade_frames.append(trades)
            equity_frames.append(equity)
            metric_rows.append(metrics)
    metrics = pd.DataFrame(metric_rows)
    selected_rows = []
    for prefix in ("stat_garch_p", "catboost_mq_p"):
        selected = select_probability_threshold(metrics, prefix, minimum_validation_trades)
        if np.isfinite(selected):
            strategy = f"{prefix}{selected:.2f}"
            selected_rows.append(
                metrics.loc[metrics["strategy"].eq(strategy) & metrics["segment"].eq("evaluation")].iloc[0].to_dict()
                | {"selected_on": "validation", "probability_threshold": selected}
            )
    return (
        pd.concat([x for x in trade_frames if not x.empty], ignore_index=True) if any(not x.empty for x in trade_frames) else pd.DataFrame(),
        pd.concat([x for x in equity_frames if not x.empty], ignore_index=True) if any(not x.empty for x in equity_frames) else pd.DataFrame(),
        metrics,
        pd.DataFrame(selected_rows),
    )


def evaluate_horizon(args: argparse.Namespace, s3: Any, spec: ExperimentSpec, run_prefix: str, logger: logging.Logger):
    dataset_prefix, target_column = TARGET_SPECS[spec.horizon]
    train, test = load_train_test(s3, args.bucket, dataset_prefix)
    baseline_key = f"dataset_target_{spec.horizon}/{args.results_subdir}/{args.baseline_run_id}/experiment_results.parquet"
    baseline_results = read_required_parquet(s3, args.bucket, baseline_key)
    logger.info("fit point horizon=%d", spec.horizon)
    point_model, garch_fit_residual, delayed_residual_prefix, predictions, point_metadata = fit_selected_point_model(
        train, test, baseline_results, spec, target_column
    )
    test_residual = predictions["y_true"].to_numpy() - predictions["point_mean"].to_numpy()
    logger.info("fit garch horizon=%d model=%s", spec.horizon, spec.garch_name)
    garch_model, sigma, nu, garch_metadata = fit_residual_garch(
        garch_fit_residual,
        delayed_residual_prefix,
        test_residual,
        spec.garch_name,
        args.garch_scale,
        args.garch_maxiter,
    )
    predictions["garch_sigma"] = sigma
    cost = 2.0 * args.fee_per_action
    predictions["garch_p_long"] = 1.0 - standardized_student_cdf(cost, predictions["point_mean"], sigma, nu)
    predictions["garch_p_short"] = standardized_student_cdf(-cost, predictions["point_mean"], sigma, nu)
    ml_dataset_prefix = ML_DATASET_PREFIXES[spec.horizon]
    if ml_dataset_prefix != dataset_prefix:
        ml_train, ml_test = load_train_test(s3, args.bucket, ml_dataset_prefix)
    else:
        ml_train, ml_test = train, test
    logger.info("fit CatBoost MultiQuantile horizon=%d dataset=%s", spec.horizon, ml_dataset_prefix)
    catboost_model, catboost_predictions, catboost_metadata = fit_catboost_multiquantile(
        ml_train, ml_test, target_column, spec.horizon, args.threads
    )
    predictions = predictions.merge(catboost_predictions, on="timestamp", how="inner", validate="one_to_one")
    predictions["catboost_p_long"] = 1.0 - quantile_cdf_at(predictions, cost)
    predictions["catboost_p_short"] = quantile_cdf_at(predictions, -cost)
    trades, equity, metrics, selected = run_backtests(
        predictions,
        horizon=spec.horizon,
        fee_per_action=args.fee_per_action,
        probability_thresholds=args.probability_threshold,
        validation_fraction=args.validation_fraction,
        minimum_validation_trades=args.minimum_validation_trades,
    )
    horizon_prefix = f"{run_prefix}/horizon_{spec.horizon}"
    upload_bytes(s3, args.bucket, f"{horizon_prefix}/models/point_model.joblib", serialize_joblib(point_model), "application/octet-stream")
    upload_bytes(s3, args.bucket, f"{horizon_prefix}/models/garch_result.joblib", serialize_joblib(garch_model), "application/octet-stream")
    upload_bytes(s3, args.bucket, f"{horizon_prefix}/models/catboost_multiquantile.joblib", serialize_joblib(catboost_model), "application/octet-stream")
    upload_json(s3, args.bucket, f"{horizon_prefix}/models/point_model_metadata.json", point_metadata)
    upload_json(s3, args.bucket, f"{horizon_prefix}/models/garch_metadata.json", garch_metadata)
    upload_json(s3, args.bucket, f"{horizon_prefix}/models/catboost_metadata.json", catboost_metadata)
    for name, frame in (("predictions", predictions), ("trades", trades), ("equity", equity), ("metrics", metrics), ("selected_thresholds", selected)):
        upload_parquet(s3, args.bucket, f"{horizon_prefix}/{name}.parquet", frame)
        if name in {"metrics", "selected_thresholds"}:
            upload_csv(s3, args.bucket, f"{horizon_prefix}/{name}.csv", frame)
    return metrics, selected


def evaluate_horizon_worker(
    args: argparse.Namespace,
    spec: ExperimentSpec,
    run_prefix: str,
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    """Process-safe horizon entry point; clients and thread pools stay local."""
    logger = setup_logging()
    s3 = make_s3_client()
    with threadpool_limits(limits=args.threads):
        metrics, selected = evaluate_horizon(args, s3, spec, run_prefix, logger)
    return spec.horizon, metrics, selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cost-aware non-overlapping trading tests for selected probabilistic models.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--results-subdir", default=DEFAULT_RESULTS_SUBDIR)
    parser.add_argument("--baseline-run-id", default="latest")
    parser.add_argument("--horizon", type=int, choices=sorted(EXPERIMENT_SPECS), action="append", default=None)
    parser.add_argument("--fee-per-action", type=float, default=DEFAULT_FEE_PER_ACTION)
    parser.add_argument("--probability-threshold", type=float, action="append", default=None)
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument("--minimum-validation-trades", type=int, default=30)
    parser.add_argument("--garch-scale", type=float, default=100.0)
    parser.add_argument("--garch-maxiter", type=int, default=1000)
    parser.add_argument(
        "--max-parallel-horizons",
        type=int,
        default=1,
        help="Separate worker processes. Use 3 to run 10m, 20m, and 30m concurrently.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="CatBoost and native math threads available inside each horizon process.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.probability_threshold is None:
        args.probability_threshold = list(DEFAULT_PROBABILITY_THRESHOLDS)
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.fee_per_action < 0:
        parser.error("--fee-per-action must be non-negative")
    if args.max_parallel_horizons < 1:
        parser.error("--max-parallel-horizons must be at least 1")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    run_id = args.run_id or make_run_id()
    horizons = sorted(set(args.horizon or EXPERIMENT_SPECS))
    config = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": args.bucket,
        "output_prefix": args.output_prefix,
        "horizons": horizons,
        "fee_per_action": args.fee_per_action,
        "round_trip_fee": 2 * args.fee_per_action,
        "probability_thresholds": args.probability_threshold,
        "validation_fraction": args.validation_fraction,
        "max_parallel_horizons": min(args.max_parallel_horizons, len(horizons)),
        "threads_per_horizon": args.threads,
        "maximum_requested_compute_threads": min(args.max_parallel_horizons, len(horizons)) * args.threads,
        "execution_mode": "target_log_return_proxy",
        "execution_note": "Target log return is used as the gross holding-period return; no slippage or funding is included.",
        "position_policy": "one non-overlapping position at a time per horizon",
        "model_specs": {h: asdict(EXPERIMENT_SPECS[h]) for h in horizons},
    }
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    s3 = make_s3_client()
    run_prefix = f"{args.output_prefix.strip('/')}/{run_id}"
    upload_json(s3, args.bucket, f"{run_prefix}/run_config.json", config)
    metric_frames, selected_frames = [], []
    started = time.perf_counter()
    workers = min(args.max_parallel_horizons, len(horizons))
    if workers == 1:
        for horizon in horizons:
            metrics, selected = evaluate_horizon(args, s3, EXPERIMENT_SPECS[horizon], run_prefix, logger)
            metric_frames.append(metrics)
            selected_frames.append(selected)
    else:
        logger.info(
            "parallel execution workers=%d threads_per_worker=%d requested_threads=%d",
            workers,
            args.threads,
            workers * args.threads,
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(evaluate_horizon_worker, args, EXPERIMENT_SPECS[horizon], run_prefix): horizon
                for horizon in horizons
            }
            for future in as_completed(futures):
                horizon = futures[future]
                completed_horizon, metrics, selected = future.result()
                metric_frames.append(metrics)
                selected_frames.append(selected)
                logger.info("completed horizon=%d", completed_horizon)
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_selected = pd.concat(selected_frames, ignore_index=True)
    upload_parquet(s3, args.bucket, f"{run_prefix}/all_metrics.parquet", all_metrics)
    upload_csv(s3, args.bucket, f"{run_prefix}/all_metrics.csv", all_metrics)
    upload_parquet(s3, args.bucket, f"{run_prefix}/selected_evaluation_results.parquet", all_selected)
    upload_csv(s3, args.bucket, f"{run_prefix}/selected_evaluation_results.csv", all_selected)
    summary = {**config, "elapsed_sec": time.perf_counter() - started, "run_uri": f"s3://{args.bucket}/{run_prefix}/"}
    upload_json(s3, args.bucket, f"{run_prefix}/run_summary.json", summary)
    upload_json(s3, args.bucket, f"{args.output_prefix.strip('/')}/latest.json", {"run_id": run_id, "run_uri": summary["run_uri"]})
    logger.info("completed run_uri=%s", summary["run_uri"])


if __name__ == "__main__":
    main()
