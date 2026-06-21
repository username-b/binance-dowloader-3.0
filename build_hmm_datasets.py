"""Build minute-level HMM observation datasets from S3 sources."""

from __future__ import annotations

import argparse
import gc
import io
from datetime import timedelta

import numpy as np
import pandas as pd

from build_price_feature_day import (
    RAW_HISTORY_START_DATE,
    kline_key,
    load_s3_parquet,
    make_s3_client,
    s3_partition_is_current,
    unified_dataset_key,
)


# Inclusive UTC range for a normal run. Edit these values before launching.
RUN_START_DATE = "2020-05-27"
RUN_END_DATE = "2023-02-01"

HMM_OUTPUT_PREFIX = "features/hmm_dataset"
HMM_SCHEMA_VERSION = "2"

RETURN_WINDOWS = (3, 5, 10, 20, 30, 60)
KAUFMAN_WINDOWS = (5, 10, 20, 30, 60, 90)
VOLATILITY_WINDOWS = (5, 10, 20, 30, 60)
ZSCORE_WINDOWS = (5, 10, 20, 30, 60, 90)
TRADE_FLOW_WINDOWS = (5, 10, 20, 30)

CORRELATION_WINDOWS = (30, 60, 180)
RETURN_DIFFERENCE_WINDOWS = (5, 10, 30, 60)
BETA_WINDOWS = (30, 60, 180)

MARKET_FEATURE_COLUMNS = tuple(
    ["ada_log_return_lag_1"]
    + [f"ada_cum_return_{window}m" for window in RETURN_WINDOWS]
    + [f"ada_kaufman_efficiency_{window}m" for window in KAUFMAN_WINDOWS]
    + [f"ada_return_std_{window}m" for window in VOLATILITY_WINDOWS]
    + [f"ada_realized_volatility_{window}m" for window in VOLATILITY_WINDOWS]
    + [f"ada_close_zscore_{window}m" for window in ZSCORE_WINDOWS]
    + [f"ada_aggression_delta_norm_{window}m" for window in TRADE_FLOW_WINDOWS]
    + [f"ada_volume_entropy_norm_{window}m" for window in TRADE_FLOW_WINDOWS]
    + [f"ada_inverse_simpson_norm_{window}m" for window in TRADE_FLOW_WINDOWS]
    + [f"ada_pressure_concentration_{window}m" for window in TRADE_FLOW_WINDOWS]
)

RELATIVE_FEATURE_COLUMNS = tuple(
    [f"ada_btc_return_corr_{window}m" for window in CORRELATION_WINDOWS]
    + [
        f"ada_minus_btc_log_return_{window}m"
        for window in RETURN_DIFFERENCE_WINDOWS
    ]
    + [f"ada_beta_to_btc_{window}m" for window in BETA_WINDOWS]
)
HMM_FEATURE_COLUMNS = (*MARKET_FEATURE_COLUMNS, *RELATIVE_FEATURE_COLUMNS)


def hmm_dataset_key(prefix: str, symbol: str, date: str) -> str:
    return (
        f"{prefix.strip('/')}/symbol={symbol}/interval=1m/"
        f"date={date}/data.parquet"
    )


def build_market_regime_dataset(unified: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", *MARKET_FEATURE_COLUMNS}
    missing = required.difference(unified.columns)
    if missing:
        raise ValueError(f"Unified dataset is missing HMM factors: {sorted(missing)}")

    result = unified[["timestamp", *MARKET_FEATURE_COLUMNS]].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    if result["timestamp"].isna().any() or result["timestamp"].duplicated().any():
        raise ValueError("Market HMM source contains invalid timestamps")

    result = result.sort_values("timestamp").dropna().reset_index(drop=True)
    for column in MARKET_FEATURE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float32")
    _validate_finite(result, MARKET_FEATURE_COLUMNS)
    return result


def _prepare_close(klines: pd.DataFrame, label: str) -> pd.Series:
    required = {"timestamp", "close"}
    missing = required.difference(klines.columns)
    if missing:
        raise ValueError(f"{label} klines are missing columns: {sorted(missing)}")

    timestamp = pd.to_datetime(klines["timestamp"], utc=True, errors="coerce")
    close = pd.to_numeric(klines["close"], errors="coerce").astype("float64")
    if timestamp.isna().any() or close.isna().any() or (close <= 0).any():
        raise ValueError(f"{label} klines contain invalid timestamp or close")

    series = pd.Series(close.to_numpy(), index=pd.DatetimeIndex(timestamp), name=label)
    if series.index.duplicated().any():
        raise ValueError(f"{label} klines contain duplicate timestamps")
    series = series.sort_index()
    complete_index = pd.date_range(
        series.index.min().floor("min"),
        series.index.max().floor("min"),
        freq="1min",
        tz="UTC",
    )
    return series.reindex(complete_index)


def build_relative_btc_dataset(
    ada_klines: pd.DataFrame,
    btc_klines: pd.DataFrame,
    date: str,
) -> pd.DataFrame:
    ada_close = _prepare_close(ada_klines, "ada_close")
    btc_close = _prepare_close(btc_klines, "btc_close")
    prices = pd.concat([ada_close, btc_close], axis=1)
    ada_log = np.log(prices["ada_close"])
    btc_log = np.log(prices["btc_close"])
    ada_1m = ada_log.diff()
    btc_1m = btc_log.diff()

    features = pd.DataFrame(index=prices.index)
    rolling_stats = {}
    for window in set((*CORRELATION_WINDOWS, *BETA_WINDOWS)):
        ada_window = ada_1m.rolling(window, min_periods=window)
        btc_window = btc_1m.rolling(window, min_periods=window)
        covariance = ada_window.cov(btc_1m)
        ada_variance = ada_window.var(ddof=1)
        btc_variance = btc_window.var(ddof=1)
        full_window = ada_window.count().eq(window) & btc_window.count().eq(window)
        rolling_stats[window] = (
            covariance,
            ada_variance,
            btc_variance,
            full_window,
        )

    for window in CORRELATION_WINDOWS:
        covariance, ada_variance, btc_variance, full_window = rolling_stats[window]
        denominator = np.sqrt(ada_variance * btc_variance)
        correlation = covariance / denominator
        # Correlation is undefined for a fully flat window. Representing it as
        # zero preserves the minute and records that there is no co-movement.
        correlation = correlation.mask(full_window & denominator.eq(0), 0.0)
        features[f"ada_btc_return_corr_{window}m"] = correlation

    for window in RETURN_DIFFERENCE_WINDOWS:
        ada_return = ada_log - ada_log.shift(window)
        btc_return = btc_log - btc_log.shift(window)
        features[f"ada_minus_btc_log_return_{window}m"] = ada_return - btc_return

    for window in BETA_WINDOWS:
        covariance, _, btc_variance, full_window = rolling_stats[window]
        beta = covariance / btc_variance
        # Beta has the same 0/0 degeneracy when BTC is flat for the whole
        # window. Zero means no measurable sensitivity to BTC in that window.
        features[f"ada_beta_to_btc_{window}m"] = beta.mask(
            full_window & btc_variance.eq(0), 0.0
        )

    day = pd.Timestamp(date, tz="UTC")
    end = day + timedelta(days=1)
    result = features.loc[(features.index >= day) & (features.index < end)].copy()
    result = result.reset_index(names="timestamp")
    result = result[["timestamp", *RELATIVE_FEATURE_COLUMNS]].dropna().reset_index(drop=True)
    for column in RELATIVE_FEATURE_COLUMNS:
        result[column] = result[column].astype("float32")
    _validate_finite(result, RELATIVE_FEATURE_COLUMNS)
    return result


def build_combined_hmm_dataset(
    market: pd.DataFrame,
    relative: pd.DataFrame,
) -> pd.DataFrame:
    result = market.merge(
        relative,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )
    result = result[["timestamp", *HMM_FEATURE_COLUMNS]].sort_values("timestamp")
    result = result.reset_index(drop=True)
    _validate_finite(result, HMM_FEATURE_COLUMNS)
    return result


def _validate_finite(dataset: pd.DataFrame, columns: tuple[str, ...]) -> None:
    values = dataset[list(columns)].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("HMM dataset contains non-finite factor values")


def _load_price_context(s3, bucket: str, symbol: str, date: str) -> pd.DataFrame:
    day = pd.Timestamp(date)
    frames = []
    for source_day in (day - timedelta(days=1), day):
        source_date = source_day.strftime("%Y-%m-%d")
        frame = load_s3_parquet(s3, bucket, kline_key(symbol, "1m", source_date))
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No {symbol} kline context for {date}")
    if date != RAW_HISTORY_START_DATE and len(frames) != 2:
        raise FileNotFoundError(f"Previous-day {symbol} kline context is missing for {date}")
    return pd.concat(frames, ignore_index=True)


def _partition_is_current(s3, bucket: str, key: str, dataset_kind: str) -> bool:
    try:
        response = s3.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    metadata = {name.lower(): value for name, value in response.get("Metadata", {}).items()}
    return (
        metadata.get("schema-version") == HMM_SCHEMA_VERSION
        and metadata.get("dataset-kind") == dataset_kind
    )


def _upload(s3, bucket: str, key: str, dataset: pd.DataFrame, dataset_kind: str) -> int:
    buffer = io.BytesIO()
    dataset.to_parquet(buffer, index=False, compression="zstd")
    payload = buffer.getvalue()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="application/vnd.apache.parquet",
        Metadata={
            "schema-version": HMM_SCHEMA_VERSION,
            "dataset-kind": dataset_kind,
            "rows": str(len(dataset)),
            "factors": str(len(dataset.columns) - 1),
            "frequency": "1m",
        },
    )
    return len(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build minute HMM datasets for an inclusive UTC date range"
    )
    parser.add_argument(
        "--date",
        help="Build one UTC date; overrides --start-date and --end-date",
    )
    parser.add_argument(
        "--start-date",
        default=RUN_START_DATE,
        help=f"Optional override (code default: {RUN_START_DATE})",
    )
    parser.add_argument(
        "--end-date",
        default=RUN_END_DATE,
        help=f"Optional override (code default: {RUN_END_DATE})",
    )
    parser.add_argument("--bucket", default="binance-data-downloader")
    parser.add_argument("--symbol", default="ADAUSDT")
    parser.add_argument("--btc-symbol", default="BTCUSDT")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.date or args.start_date).date()
    end = pd.Timestamp(args.date or args.end_date).date()
    if start > end:
        raise ValueError("start-date must not be later than end-date")

    s3 = make_s3_client()
    dates = pd.date_range(start, end, freq="D")
    print(f"Range: {start} -> {end} ({len(dates)} days)")
    print(f"Destination: s3://{args.bucket}/{HMM_OUTPUT_PREFIX}/")
    if start.isoformat() != RAW_HISTORY_START_DATE:
        previous = start - timedelta(days=1)
        print(f"Previous-day context for first task date: {previous}")
    uploaded = skipped = 0
    failures = []
    for position, day in enumerate(dates, start=1):
        unified = market = ada = btc = relative = combined = None
        date = day.strftime("%Y-%m-%d")
        label = f"[{position}/{len(dates)}] {date}"
        destination_key = hmm_dataset_key(HMM_OUTPUT_PREFIX, args.symbol, date)
        ready = not args.overwrite and _partition_is_current(
            s3, args.bucket, destination_key, "combined_hmm"
        )
        if ready:
            skipped += 1
            print(f"{label} skip exists")
            continue

        try:
            source_key = unified_dataset_key(args.symbol, "1m", date)
            if not s3_partition_is_current(s3, args.bucket, source_key):
                raise ValueError(f"Unified source is missing or stale: {source_key}")
            unified = load_s3_parquet(s3, args.bucket, source_key)
            market = build_market_regime_dataset(unified)
            ada = _load_price_context(s3, args.bucket, args.symbol, date)
            btc = _load_price_context(s3, args.bucket, args.btc_symbol, date)
            relative = build_relative_btc_dataset(ada, btc, date)
            combined = build_combined_hmm_dataset(market, relative)

            expected_rows = 1260 if date == RAW_HISTORY_START_DATE else 1440
            if len(combined) != expected_rows:
                raise ValueError(
                    f"Combined HMM rows for {date}: {len(combined)}, expected {expected_rows}"
                )
            size = _upload(
                s3,
                args.bucket,
                destination_key,
                combined,
                "combined_hmm",
            )

            uploaded += 1
            print(
                f"{label} uploaded rows={len(combined)} factors={len(HMM_FEATURE_COLUMNS)} "
                f"size_mib={size / 1024**2:.2f}"
            )
        except Exception as exc:
            failures.append((date, str(exc)))
            print(f"{label} ERROR: {exc}")
            if not args.continue_on_error:
                raise
        finally:
            unified = market = ada = btc = relative = combined = None
            gc.collect()

    print(f"Completed: uploaded={uploaded}, skipped={skipped}, failed={len(failures)}")
    if failures:
        for date, message in failures:
            print(f"FAILED {date}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
