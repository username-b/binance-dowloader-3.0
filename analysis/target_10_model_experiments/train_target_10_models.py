from __future__ import annotations

import io
import json
import math
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


TARGET_COLUMN = "target_log_return_10m"
TIMESTAMP_COLUMN = "timestamp"
DEFAULT_BUCKET = "binance-data-downloader"
DEFAULT_DATASET_PREFIX = "dataset_target_10"
DEFAULT_RESULTS_SUBDIR = "model_experiments"
DEFAULT_DIRECTION_THRESHOLD = 0.0025


FEATURE_BLOCKS: dict[str, dict[str, list[str]]] = {
    "ar": {
        "ar_1": ["ada_log_return_lag_1"],
        "ar_3": [
            "ada_log_return_lag_1",
            "ada_log_return_lag_2",
            "ada_log_return_lag_3",
        ],
        "ar_5": [
            "ada_log_return_lag_1",
            "ada_log_return_lag_2",
            "ada_log_return_lag_3",
            "ada_log_return_lag_5",
        ],
        "ar_10": [
            "ada_log_return_lag_1",
            "ada_log_return_lag_2",
            "ada_log_return_lag_3",
            "ada_log_return_lag_5",
            "ada_log_return_lag_10",
        ],
        "ar_all": [
            "ada_log_return_lag_1",
            "ada_log_return_lag_2",
            "ada_log_return_lag_3",
            "ada_log_return_lag_5",
            "ada_log_return_lag_10",
            "ada_log_return_lag_15",
            "ada_log_return_lag_30",
            "ada_log_return_lag_60",
        ],
    },
    "price_volatility": {
        "trend": ["ada_cum_return_30m", "ada_cum_return_60m"],
        "acceleration": [
            "ada_return_acceleration_3m",
            "ada_return_acceleration_5m",
            "ada_return_acceleration_30m",
        ],
        "efficiency": [
            "ada_kaufman_efficiency_5m",
            "ada_kaufman_efficiency_90m",
        ],
        "price_deviation": ["ada_close_zscore_30m", "ada_close_zscore_90m"],
        "volatility": [
            "ada_return_std_20m",
            "ada_return_std_60m",
            "ada_realized_volatility_5m",
            "ada_realized_volatility_20m",
            "ada_realized_volatility_60m",
        ],
        "trend_volatility": [
            "ada_cum_return_30m",
            "ada_cum_return_60m",
            "ada_return_std_20m",
            "ada_return_std_60m",
            "ada_realized_volatility_20m",
            "ada_realized_volatility_60m",
        ],
        "trend_volatility_efficiency": [
            "ada_cum_return_30m",
            "ada_cum_return_60m",
            "ada_return_std_20m",
            "ada_return_std_60m",
            "ada_realized_volatility_20m",
            "ada_realized_volatility_60m",
            "ada_kaufman_efficiency_5m",
            "ada_kaufman_efficiency_90m",
        ],
        "candle_geometry": [
            "ada_upper_wick_5m",
            "ada_upper_wick_30m",
            "ada_upper_wick_60m",
            "ada_lower_wick_5m",
            "ada_lower_wick_60m",
            "ada_log_range_20m",
            "ada_log_range_60m",
            "ada_close_position_10m",
            "ada_close_position_30m",
            "ada_close_position_60m",
            "ada_high_close_ratio_20m",
            "ada_high_close_ratio_60m",
        ],
        "all_price_volatility": [
            "ada_cum_return_30m",
            "ada_cum_return_60m",
            "ada_return_acceleration_3m",
            "ada_return_acceleration_5m",
            "ada_return_acceleration_30m",
            "ada_kaufman_efficiency_5m",
            "ada_kaufman_efficiency_90m",
            "ada_close_zscore_30m",
            "ada_close_zscore_90m",
            "ada_return_std_20m",
            "ada_return_std_60m",
            "ada_realized_volatility_5m",
            "ada_realized_volatility_20m",
            "ada_realized_volatility_60m",
            "ada_upper_wick_5m",
            "ada_upper_wick_30m",
            "ada_upper_wick_60m",
            "ada_lower_wick_5m",
            "ada_lower_wick_60m",
            "ada_log_range_20m",
            "ada_log_range_60m",
            "ada_close_position_10m",
            "ada_close_position_30m",
            "ada_close_position_60m",
            "ada_high_close_ratio_20m",
            "ada_high_close_ratio_60m",
        ],
    },
    "microstructure": {
        "volume_only": [
            "ada_volume_sum_10m",
            "ada_volume_sum_20m",
            "ada_volume_sum_30m",
            "ada_volume_sum_60m",
            "ada_volume_zscore_60m",
        ],
        "trades_only": [
            "ada_trades_per_minute_5m",
            "ada_trades_per_minute_30m",
        ],
        "aggression_only": [
            "ada_aggression_delta_norm_5m",
            "ada_aggression_delta_norm_10m",
            "ada_aggression_delta_norm_20m",
            "ada_large_trade_aggression_delta_norm_5m",
            "ada_large_trade_aggression_delta_norm_20m",
        ],
        "concentration_only": [
            "ada_volume_entropy_norm_5m",
            "ada_inverse_simpson_norm_30m",
            "ada_pressure_concentration_5m",
            "ada_pressure_concentration_30m",
        ],
        "trade_size_only": [
            "ada_average_trade_size_quote_10m",
            "ada_average_trade_size_quote_30m",
        ],
        "volume_trades": [
            "ada_volume_sum_10m",
            "ada_volume_sum_20m",
            "ada_volume_sum_30m",
            "ada_volume_sum_60m",
            "ada_volume_zscore_60m",
            "ada_trades_per_minute_5m",
            "ada_trades_per_minute_30m",
        ],
        "aggression_concentration": [
            "ada_aggression_delta_norm_5m",
            "ada_aggression_delta_norm_10m",
            "ada_aggression_delta_norm_20m",
            "ada_large_trade_aggression_delta_norm_5m",
            "ada_large_trade_aggression_delta_norm_20m",
            "ada_volume_entropy_norm_5m",
            "ada_inverse_simpson_norm_30m",
            "ada_pressure_concentration_5m",
            "ada_pressure_concentration_30m",
        ],
        "all_microstructure": [
            "ada_volume_sum_10m",
            "ada_volume_sum_20m",
            "ada_volume_sum_30m",
            "ada_volume_sum_60m",
            "ada_volume_zscore_60m",
            "ada_trades_per_minute_5m",
            "ada_trades_per_minute_30m",
            "ada_aggression_delta_norm_5m",
            "ada_aggression_delta_norm_10m",
            "ada_aggression_delta_norm_20m",
            "ada_large_trade_aggression_delta_norm_5m",
            "ada_large_trade_aggression_delta_norm_20m",
            "ada_volume_entropy_norm_5m",
            "ada_inverse_simpson_norm_30m",
            "ada_average_trade_size_quote_10m",
            "ada_average_trade_size_quote_30m",
            "ada_pressure_concentration_5m",
            "ada_pressure_concentration_30m",
        ],
    },
    "btc": {
        "btc_returns_only": ["btc_log_return_30m", "btc_log_return_60m"],
        "btc_volatility_only": ["btc_return_std_30m"],
        "btc_efficiency_only": ["btc_kaufman_efficiency_60m"],
        "btc_returns_volatility": [
            "btc_log_return_30m",
            "btc_log_return_60m",
            "btc_return_std_30m",
        ],
        "all_btc": [
            "btc_log_return_30m",
            "btc_log_return_60m",
            "btc_return_std_30m",
            "btc_kaufman_efficiency_60m",
        ],
    },
    "calendar_sessions": {
        "time_of_day": ["time_of_day_sin", "time_of_day_cos"],
        "day_of_week": ["day_of_week_sin", "day_of_week_cos", "is_weekend"],
        "sessions": [
            "is_asia_open",
            "is_cboe_europe_open",
            "is_europe_open",
            "is_us_open",
        ],
        "time_sessions": [
            "time_of_day_sin",
            "time_of_day_cos",
            "is_asia_open",
            "is_cboe_europe_open",
            "is_europe_open",
            "is_us_open",
        ],
        "day_sessions": [
            "day_of_week_sin",
            "day_of_week_cos",
            "is_weekend",
            "is_asia_open",
            "is_cboe_europe_open",
            "is_europe_open",
            "is_us_open",
        ],
        "all_calendar_sessions": [
            "time_of_day_sin",
            "time_of_day_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "is_weekend",
            "is_asia_open",
            "is_cboe_europe_open",
            "is_europe_open",
            "is_us_open",
        ],
    },
}


STAGES = [
    ("stage_0_baseline", "baseline"),
    ("stage_1_ar", "ar"),
    ("stage_2_price_volatility", "price_volatility"),
    ("stage_3_microstructure", "microstructure"),
    ("stage_4_btc", "btc"),
    ("stage_5_calendar_sessions", "calendar_sessions"),
]


@dataclass(frozen=True)
class ModelFamily:
    name: str
    simplicity_rank: int
    estimator: Any


@dataclass(frozen=True)
class PreparedData:
    feature_columns: list[str]
    feature_index: dict[str, int]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    baseline_zero_sse: float
    prep_time: float


MODEL_FAMILIES = [
    ModelFamily("ols", 10, LinearRegression()),
    ModelFamily("ridge_alpha_0_1", 20, Ridge(alpha=0.1, random_state=42)),
    ModelFamily("ridge_alpha_1", 21, Ridge(alpha=1.0, random_state=42)),
    ModelFamily("ridge_alpha_10", 22, Ridge(alpha=10.0, random_state=42)),
    ModelFamily("lasso_alpha_1e_5", 30, Lasso(alpha=1e-5, max_iter=5000, random_state=42)),
    ModelFamily("lasso_alpha_1e_4", 31, Lasso(alpha=1e-4, max_iter=5000, random_state=42)),
    ModelFamily(
        "elasticnet_alpha_1e_4_l1_0_5",
        40,
        ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000, random_state=42),
    ),
    ModelFamily("huber", 50, HuberRegressor(max_iter=1000)),
]


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "build_price_feature_day.py").exists():
            return candidate
    raise FileNotFoundError("Could not find project root with build_price_feature_day.py")


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_target_data(s3, bucket: str, dataset_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from build_price_feature_day import load_s3_parquet

    train_key = f"{dataset_prefix.strip('/')}/train.parquet"
    test_key = f"{dataset_prefix.strip('/')}/test.parquet"
    train = load_s3_parquet(s3, bucket, train_key)
    test = load_s3_parquet(s3, bucket, test_key)
    if train is None:
        raise FileNotFoundError(f"s3://{bucket}/{train_key}")
    if test is None:
        raise FileNotFoundError(f"s3://{bucket}/{test_key}")
    return train, test


def upload_bytes(s3, bucket: str, key: str, payload: bytes, content_type: str) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)


def upload_dataframe_parquet(s3, bucket: str, key: str, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, engine="pyarrow", compression="zstd")
    upload_bytes(
        s3,
        bucket,
        key,
        buffer.getvalue(),
        "application/vnd.apache.parquet",
    )


def upload_dataframe_csv(s3, bucket: str, key: str, frame: pd.DataFrame) -> None:
    payload = frame.to_csv(index=False).encode("utf-8-sig")
    upload_bytes(s3, bucket, key, payload, "text/csv; charset=utf-8")


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def available_features(columns: set[str], features: list[str]) -> list[str]:
    return [feature for feature in dedupe_preserve_order(features) if feature in columns]


def prepare_target(frame: pd.DataFrame, target_column: str) -> pd.Series:
    target = pd.to_numeric(frame[target_column], errors="coerce")
    return target.replace([np.inf, -np.inf], np.nan)


def selected_experiment_features(columns: set[str]) -> list[str]:
    features = []
    for block in FEATURE_BLOCKS.values():
        for block_features in block.values():
            features.extend(block_features)
    return available_features(columns, features)


def prepare_experiment_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str,
) -> PreparedData:
    started = time.perf_counter()
    common_columns = set(train.columns).intersection(test.columns)
    feature_columns = selected_experiment_features(common_columns)
    if target_column not in train.columns or target_column not in test.columns:
        raise ValueError(f"Target column is missing: {target_column}")

    y_train = prepare_target(train, target_column)
    y_test = prepare_target(test, target_column)
    train_mask = y_train.notna().to_numpy()
    test_mask = y_test.notna().to_numpy()

    X_train_raw = (
        train.loc[train_mask, feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float64, copy=True)
    )
    X_test_raw = (
        test.loc[test_mask, feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float64, copy=True)
    )

    medians = np.nanmedian(X_train_raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train_nan_rows, train_nan_cols = np.where(~np.isfinite(X_train_raw))
    test_nan_rows, test_nan_cols = np.where(~np.isfinite(X_test_raw))
    X_train_raw[train_nan_rows, train_nan_cols] = medians[train_nan_cols]
    X_test_raw[test_nan_rows, test_nan_cols] = medians[test_nan_cols]

    means = X_train_raw.mean(axis=0)
    stds = X_train_raw.std(axis=0)
    stds = np.where(stds > 0, stds, 1.0)
    X_train = (X_train_raw - means) / stds
    X_test = (X_test_raw - means) / stds

    y_train_array = y_train.loc[train_mask].to_numpy(dtype=np.float64)
    y_test_array = y_test.loc[test_mask].to_numpy(dtype=np.float64)
    return PreparedData(
        feature_columns=feature_columns,
        feature_index={feature: idx for idx, feature in enumerate(feature_columns)},
        X_train=X_train,
        X_test=X_test,
        y_train=y_train_array,
        y_test=y_test_array,
        baseline_zero_sse=float(np.square(y_test_array).sum()),
        prep_time=time.perf_counter() - started,
    )


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0) & (y_pred != 0)
    if not valid.any():
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


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2:
        return np.nan
    if np.nanstd(y_pred[valid]) == 0 or np.nanstd(y_true[valid]) == 0:
        return np.nan
    return float(np.corrcoef(y_true[valid], y_pred[valid])[0, 1])


def build_estimator(family: ModelFamily, n_features: int) -> Any:
    if n_features == 0:
        if family.name == "baseline_zero":
            return DummyRegressor(strategy="constant", constant=0.0)
        return DummyRegressor(strategy="mean")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clone(family.estimator)),
        ]
    )


def subset_matrix(matrix: np.ndarray, feature_indices: list[int]) -> np.ndarray:
    if not feature_indices:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    return matrix[:, feature_indices]


def predict_baseline(family: ModelFamily, y_train: np.ndarray, n_test: int) -> tuple[np.ndarray, float, float]:
    started = time.perf_counter()
    if family.name == "baseline_zero":
        value = 0.0
    else:
        value = float(np.mean(y_train))
    fit_time = time.perf_counter() - started
    started = time.perf_counter()
    y_pred = np.full(n_test, value, dtype=np.float64)
    predict_time = time.perf_counter() - started
    return y_pred, fit_time, predict_time


def fit_predict_prepared(
    data: PreparedData,
    family: ModelFamily,
    feature_indices: list[int],
) -> tuple[np.ndarray, float, float]:
    if not feature_indices:
        return predict_baseline(family, data.y_train, len(data.y_test))

    X_train = subset_matrix(data.X_train, feature_indices)
    X_test = subset_matrix(data.X_test, feature_indices)
    estimator = clone(family.estimator)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        with threadpool_limits(limits=1):
            estimator.fit(X_train, data.y_train)
    fit_time = time.perf_counter() - started

    started = time.perf_counter()
    with threadpool_limits(limits=1):
        y_pred = estimator.predict(X_test)
    predict_time = time.perf_counter() - started
    return y_pred, fit_time, predict_time


def result_row_from_predictions(
    *,
    data: PreparedData,
    y_pred: np.ndarray,
    fit_time: float,
    predict_time: float,
    model_id: str,
    stage: str,
    model_family: ModelFamily,
    model_name: str,
    feature_set: str,
    features: list[str],
    previous_model: str | None,
    added_block: str,
    direction_threshold: float,
    baseline_metrics: dict[str, float] | None,
    target_column: str,
) -> dict[str, Any]:
    y_test_array = data.y_test
    residual = y_test_array - y_pred
    sse = float(np.square(residual).sum())
    rmse = float(math.sqrt(mean_squared_error(y_test_array, y_pred)))
    da = direction_accuracy(y_test_array, y_pred)
    da_025, da_025_coverage, da_025_count = threshold_direction_accuracy(
        y_test_array,
        y_pred,
        direction_threshold,
    )

    row: dict[str, Any] = {
        "model_id": model_id,
        "stage": stage,
        "model_family": model_family.name,
        "model_name": model_name,
        "feature_set": feature_set,
        "features": json.dumps(features, ensure_ascii=False),
        "n_features": len(features),
        "train_size": int(len(data.y_train)),
        "test_size": int(len(data.y_test)),
        "fit_time": fit_time,
        "predict_time": predict_time,
        "MAE": float(mean_absolute_error(y_test_array, y_pred)),
        "RMSE": rmse,
        "Direction_Accuracy": da,
        "Direction_Accuracy_025": da_025,
        "DA_025_coverage": da_025_coverage,
        "DA_025_count": da_025_count,
        "R2": float(r2_score(y_test_array, y_pred)),
        "OOS_R2": float(1.0 - sse / data.baseline_zero_sse) if data.baseline_zero_sse > 0 else np.nan,
        "Pearson_corr": pearson_corr(y_test_array, y_pred),
        "Mean_residual": float(np.mean(residual)),
        "Std_residual": float(np.std(residual)),
        "Max_absolute_error": float(np.max(np.abs(residual))),
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
        "true_mean": float(np.mean(y_test_array)),
        "true_std": float(np.std(y_test_array)),
        "previous_model": previous_model,
        "added_block": added_block,
        "delta_MAE": np.nan,
        "delta_RMSE": np.nan,
        "delta_DA": np.nan,
        "delta_DA_025": np.nan,
        "delta_features": np.nan,
        "simplicity_rank": model_family.simplicity_rank,
        "direction_threshold": direction_threshold,
        "target_name": target_column,
        "rank_A": np.nan,
        "rank_B": np.nan,
        "selected_for_next_stage": False,
        "selection_reason": "",
        "prep_time": data.prep_time,
    }

    if model_family.name == "baseline_zero":
        row["OOS_R2"] = 0.0

    if baseline_metrics:
        row["delta_MAE"] = row["MAE"] - baseline_metrics.get("MAE", np.nan)
        row["delta_RMSE"] = row["RMSE"] - baseline_metrics.get("RMSE", np.nan)
        row["delta_DA"] = row["Direction_Accuracy"] - baseline_metrics.get("Direction_Accuracy", np.nan)
        row["delta_DA_025"] = row["Direction_Accuracy_025"] - baseline_metrics.get(
            "Direction_Accuracy_025",
            np.nan,
        )
        row["delta_features"] = row["n_features"] - baseline_metrics.get("n_features", np.nan)
    return row


def evaluate_prepared_model(
    data: PreparedData,
    *,
    model_id: str,
    stage: str,
    model_family: ModelFamily,
    model_name: str,
    feature_set: str,
    features: list[str],
    previous_model: str | None,
    added_block: str,
    direction_threshold: float,
    baseline_metrics: dict[str, float] | None,
    target_column: str,
) -> dict[str, Any]:
    features = dedupe_preserve_order(features)
    feature_indices = [data.feature_index[feature] for feature in features]
    y_pred, fit_time, predict_time = fit_predict_prepared(data, model_family, feature_indices)
    return result_row_from_predictions(
        data=data,
        y_pred=y_pred,
        fit_time=fit_time,
        predict_time=predict_time,
        model_id=model_id,
        stage=stage,
        model_family=model_family,
        model_name=model_name,
        feature_set=feature_set,
        features=features,
        previous_model=previous_model,
        added_block=added_block,
        direction_threshold=direction_threshold,
        baseline_metrics=baseline_metrics,
        target_column=target_column,
    )


def evaluate_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    model_id: str,
    stage: str,
    model_family: ModelFamily,
    model_name: str,
    feature_set: str,
    features: list[str],
    previous_model: str | None,
    added_block: str,
    direction_threshold: float,
    baseline_sse: float,
    baseline_metrics: dict[str, float] | None,
    target_column: str,
) -> tuple[dict[str, Any], np.ndarray]:
    features = dedupe_preserve_order(features)
    columns = features + [target_column]
    train_part = train.loc[:, columns].copy()
    test_part = test.loc[:, columns].copy()
    train_part = train_part.replace([np.inf, -np.inf], np.nan)
    test_part = test_part.replace([np.inf, -np.inf], np.nan)

    y_train = prepare_target(train_part, target_column)
    y_test = prepare_target(test_part, target_column)
    train_mask = y_train.notna()
    test_mask = y_test.notna()

    X_train = train_part.loc[train_mask, features] if features else np.empty((int(train_mask.sum()), 0))
    X_test = test_part.loc[test_mask, features] if features else np.empty((int(test_mask.sum()), 0))
    y_train_array = y_train.loc[train_mask].to_numpy(dtype=float)
    y_test_array = y_test.loc[test_mask].to_numpy(dtype=float)

    estimator = build_estimator(model_family, len(features))
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(X_train, y_train_array)
    fit_time = time.perf_counter() - started

    started = time.perf_counter()
    y_pred = estimator.predict(X_test)
    predict_time = time.perf_counter() - started

    residual = y_test_array - y_pred
    sse = float(np.square(residual).sum())
    rmse = float(math.sqrt(mean_squared_error(y_test_array, y_pred)))
    da = direction_accuracy(y_test_array, y_pred)
    da_025, da_025_coverage, da_025_count = threshold_direction_accuracy(
        y_test_array,
        y_pred,
        direction_threshold,
    )

    row: dict[str, Any] = {
        "model_id": model_id,
        "stage": stage,
        "model_family": model_family.name,
        "model_name": model_name,
        "feature_set": feature_set,
        "features": json.dumps(features, ensure_ascii=False),
        "n_features": len(features),
        "train_size": int(len(y_train_array)),
        "test_size": int(len(y_test_array)),
        "fit_time": fit_time,
        "predict_time": predict_time,
        "MAE": float(mean_absolute_error(y_test_array, y_pred)),
        "RMSE": rmse,
        "Direction_Accuracy": da,
        "Direction_Accuracy_025": da_025,
        "DA_025_coverage": da_025_coverage,
        "DA_025_count": da_025_count,
        "R2": float(r2_score(y_test_array, y_pred)),
        "OOS_R2": float(1.0 - sse / baseline_sse) if baseline_sse > 0 else np.nan,
        "Pearson_corr": pearson_corr(y_test_array, y_pred),
        "Mean_residual": float(np.mean(residual)),
        "Std_residual": float(np.std(residual)),
        "Max_absolute_error": float(np.max(np.abs(residual))),
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
        "true_mean": float(np.mean(y_test_array)),
        "true_std": float(np.std(y_test_array)),
        "previous_model": previous_model,
        "added_block": added_block,
        "delta_MAE": np.nan,
        "delta_RMSE": np.nan,
        "delta_DA": np.nan,
        "delta_DA_025": np.nan,
        "delta_features": np.nan,
        "simplicity_rank": model_family.simplicity_rank,
        "direction_threshold": direction_threshold,
        "target_name": target_column,
        "rank_A": np.nan,
        "rank_B": np.nan,
        "selected_for_next_stage": False,
        "selection_reason": "",
    }

    if baseline_metrics:
        row["delta_MAE"] = row["MAE"] - baseline_metrics.get("MAE", np.nan)
        row["delta_RMSE"] = row["RMSE"] - baseline_metrics.get("RMSE", np.nan)
        row["delta_DA"] = row["Direction_Accuracy"] - baseline_metrics.get("Direction_Accuracy", np.nan)
        row["delta_DA_025"] = row["Direction_Accuracy_025"] - baseline_metrics.get(
            "Direction_Accuracy_025",
            np.nan,
        )
        row["delta_features"] = row["n_features"] - baseline_metrics.get("n_features", np.nan)

    return row, y_pred


def rank_and_select(stage_results: pd.DataFrame, max_selected: int = 6) -> tuple[pd.DataFrame, list[str]]:
    ranked = stage_results.copy()
    rank_a = (
        ranked.sort_values(
            ["MAE", "RMSE", "n_features", "simplicity_rank", "model_id"],
            ascending=[True, True, True, True, True],
        )["model_id"]
        .reset_index(drop=True)
        .reset_index()
    )
    rank_a_map = dict(zip(rank_a["model_id"], rank_a["index"] + 1))
    rank_b = (
        ranked.sort_values(
            ["Direction_Accuracy_025", "n_features", "simplicity_rank", "model_id"],
            ascending=[False, True, True, True],
            na_position="last",
        )["model_id"]
        .reset_index(drop=True)
        .reset_index()
    )
    rank_b_map = dict(zip(rank_b["model_id"], rank_b["index"] + 1))
    ranked["rank_A"] = ranked["model_id"].map(rank_a_map)
    ranked["rank_B"] = ranked["model_id"].map(rank_b_map)

    top_a = ranked.nsmallest(3, "rank_A")["model_id"].tolist()
    top_b = ranked.nsmallest(3, "rank_B")["model_id"].tolist()
    selected = dedupe_preserve_order([*top_a, *top_b])[:max_selected]

    ranked["selected_for_next_stage"] = ranked["model_id"].isin(selected)
    ranked.loc[ranked["model_id"].isin(top_a), "selection_reason"] = "top_A"
    ranked.loc[ranked["model_id"].isin(top_b), "selection_reason"] = ranked.loc[
        ranked["model_id"].isin(top_b),
        "selection_reason",
    ].replace("", "top_B")
    both = ranked["model_id"].isin(set(top_a).intersection(top_b))
    ranked.loc[both, "selection_reason"] = "top_A_top_B"
    return ranked, selected


def make_baseline_families() -> list[ModelFamily]:
    return [
        ModelFamily("baseline_zero", 0, None),
        ModelFamily("baseline_mean", 1, None),
    ]


def generate_stage_candidates(
    *,
    stage_name: str,
    block_name: str,
    previous_rows: pd.DataFrame | None,
    available_columns: set[str],
) -> list[dict[str, Any]]:
    if block_name == "baseline":
        return [
            {
                "stage": stage_name,
                "feature_set": "baseline_zero",
                "features": [],
                "previous_model": None,
                "added_block": "baseline_zero",
            },
            {
                "stage": stage_name,
                "feature_set": "baseline_mean",
                "features": [],
                "previous_model": None,
                "added_block": "baseline_mean",
            },
        ]

    block_variants = FEATURE_BLOCKS[block_name]
    bases = [{"model_id": None, "feature_set": "", "features": "[]"}]
    if previous_rows is not None and not previous_rows.empty:
        bases = previous_rows.to_dict("records")

    candidates = []
    seen = set()
    for base in bases:
        base_features = json.loads(base.get("features") or "[]")
        for variant_name, variant_features in block_variants.items():
            added = available_features(available_columns, variant_features)
            features = dedupe_preserve_order([*base_features, *added])
            if not features:
                continue
            feature_key = tuple(features)
            key = (stage_name, variant_name, feature_key)
            if key in seen:
                continue
            seen.add(key)
            base_set = base.get("feature_set") or "root"
            candidates.append(
                {
                    "stage": stage_name,
                    "feature_set": f"{base_set}+{variant_name}",
                    "features": features,
                    "previous_model": base.get("model_id"),
                    "added_block": variant_name,
                }
            )
    return candidates


def run_experiments(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    direction_threshold: float = DEFAULT_DIRECTION_THRESHOLD,
    max_selected_per_stage: int = 6,
    n_jobs: int = 1,
    parallel_backend: str = "threading",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_columns = set(train.columns).intersection(test.columns)
    data = prepare_experiment_data(train, test, target_column=target_column)
    results: list[dict[str, Any]] = []
    selected_rows: pd.DataFrame | None = None
    metrics_by_id: dict[str, dict[str, float]] = {}
    stage_selection: list[dict[str, Any]] = []

    for stage_index, (stage_name, block_name) in enumerate(STAGES):
        candidates = generate_stage_candidates(
            stage_name=stage_name,
            block_name=block_name,
            previous_rows=selected_rows,
            available_columns=available_columns,
        )
        families = make_baseline_families() if block_name == "baseline" else MODEL_FAMILIES
        jobs = []

        for candidate_index, candidate in enumerate(candidates, start=1):
            for family in families:
                if block_name == "baseline" and family.name not in candidate["feature_set"]:
                    continue
                model_id = f"{stage_name}_{candidate_index:03d}_{family.name}"
                previous_model = candidate["previous_model"]
                previous_metrics = metrics_by_id.get(previous_model or "", None)
                jobs.append(
                    {
                        "model_id": model_id,
                        "stage": stage_name,
                        "model_family": family,
                        "model_name": f"{candidate['feature_set']}__{family.name}",
                        "feature_set": candidate["feature_set"],
                        "features": candidate["features"],
                        "previous_model": previous_model,
                        "added_block": candidate["added_block"],
                        "direction_threshold": direction_threshold,
                        "baseline_metrics": previous_metrics,
                        "target_column": target_column,
                    }
                )

        if n_jobs == 1 or len(jobs) <= 1:
            stage_rows = [evaluate_prepared_model(data, **job) for job in jobs]
        else:
            stage_rows = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
                delayed(evaluate_prepared_model)(data, **job) for job in jobs
            )

        for row in stage_rows:
            metrics_by_id[row["model_id"]] = {
                "MAE": row["MAE"],
                "RMSE": row["RMSE"],
                "Direction_Accuracy": row["Direction_Accuracy"],
                "Direction_Accuracy_025": row["Direction_Accuracy_025"],
                "n_features": row["n_features"],
            }

        stage_frame, selected_ids = rank_and_select(
            pd.DataFrame(stage_rows),
            max_selected=max_selected_per_stage,
        )
        results.extend(stage_frame.to_dict("records"))
        selected_rows = stage_frame.loc[stage_frame["model_id"].isin(selected_ids)].copy()
        stage_selection.append(
            {
                "stage": stage_name,
                "stage_index": stage_index,
                "candidates": len(stage_frame),
                "selected": len(selected_ids),
                "selected_model_ids": json.dumps(selected_ids, ensure_ascii=False),
                "n_jobs": n_jobs,
                "parallel_backend": parallel_backend,
                "prep_time": data.prep_time,
                "prepared_features": len(data.feature_columns),
            }
        )

    return pd.DataFrame(results), pd.DataFrame(stage_selection)


def save_outputs_to_s3(
    s3,
    bucket: str,
    dataset_prefix: str,
    results_subdir: str,
    run_id: str,
    results: pd.DataFrame,
    selection: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, str]:
    output_prefix = f"{dataset_prefix.strip('/')}/{results_subdir.strip('/')}/{run_id}"
    latest_prefix = f"{dataset_prefix.strip('/')}/{results_subdir.strip('/')}/latest"
    artifacts = {
        f"{output_prefix}/experiment_results.parquet": ("parquet", results),
        f"{output_prefix}/experiment_results.csv": ("csv", results),
        f"{output_prefix}/stage_selection.parquet": ("parquet", selection),
        f"{output_prefix}/stage_selection.csv": ("csv", selection),
        f"{latest_prefix}/experiment_results.parquet": ("parquet", results),
        f"{latest_prefix}/stage_selection.parquet": ("parquet", selection),
    }
    for key, (kind, frame) in artifacts.items():
        if kind == "parquet":
            upload_dataframe_parquet(s3, bucket, key, frame)
        else:
            upload_dataframe_csv(s3, bucket, key, frame)

    config_payload = json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8")
    upload_bytes(s3, bucket, f"{output_prefix}/run_config.json", config_payload, "application/json")
    upload_bytes(s3, bucket, f"{latest_prefix}/run_config.json", config_payload, "application/json")

    return {
        "run_prefix": f"s3://{bucket}/{output_prefix}/",
        "latest_prefix": f"s3://{bucket}/{latest_prefix}/",
    }
