"""Prepare minute-level ADA features for the selected GaussianHMM model."""

from __future__ import annotations

import argparse
import io
import os
from datetime import timedelta

import numpy as np
import pandas as pd

from build_price_feature_day import kline_key, load_s3_parquet, make_s3_client


HMM_FEATURE_COLUMNS = (
    "ada_cum_return_12h",
    "ada_realized_volatility_12h",
    "ada_kaufman_efficiency_24h",
    "ada_close_position_12h",
)

TWELVE_HOURS = 12 * 60
TWENTY_FOUR_HOURS = 24 * 60
DEFAULT_BUCKET = os.getenv("YC_BUCKET", "binance-data-downloader")
DEFAULT_OUTPUT_KEY = (
    "features/hmm_dataset/ada_hmm_12h_24h/interval=1m/data.parquet"
)


def _prepare_minute_klines(klines: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(klines.columns)
    if missing:
        raise ValueError(f"Missing required kline columns: {sorted(missing)}")

    frame = klines.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")

    if frame["timestamp"].isna().any():
        raise ValueError("Klines contain invalid timestamps")
    if frame["timestamp"].duplicated().any():
        duplicates = frame.loc[frame["timestamp"].duplicated(), "timestamp"].head().tolist()
        raise ValueError(f"Klines contain duplicate timestamps: {duplicates}")
    if (frame[["open", "high", "low", "close"]].dropna() <= 0).any().any():
        raise ValueError("OHLC prices must be positive")

    valid_ohlc = frame[["open", "high", "low", "close"]].dropna()
    invalid = (
        valid_ohlc["high"].lt(valid_ohlc[["open", "close"]].max(axis=1))
        | valid_ohlc["low"].gt(valid_ohlc[["open", "close"]].min(axis=1))
        | valid_ohlc["high"].lt(valid_ohlc["low"])
    )
    if invalid.any():
        raise ValueError("Klines contain inconsistent OHLC prices")

    frame = frame.sort_values("timestamp").set_index("timestamp")
    if frame.empty:
        return frame

    complete_index = pd.date_range(
        start=frame.index.min().floor("min"),
        end=frame.index.max().floor("min"),
        freq="1min",
        tz="UTC",
        name="timestamp",
    )
    return frame.reindex(complete_index)


def build_ada_hmm_12h_24h_dataset(
    klines: pd.DataFrame,
    *,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build the four selected causal minute features from ADA 1m klines.

    The output feature names match the requested model input:
    ``ada_cum_return_12h``, ``ada_realized_volatility_12h``,
    ``ada_kaufman_efficiency_24h``, and ``ada_close_position_12h``.
    """

    frame = _prepare_minute_klines(klines)
    features = pd.DataFrame(index=frame.index)
    if frame.empty:
        return features.reset_index()[["timestamp", *HMM_FEATURE_COLUMNS]]

    log_close = np.log(frame["close"])
    minute_return = log_close.diff()

    features["ada_cum_return_12h"] = log_close - log_close.shift(TWELVE_HOURS)
    features["ada_realized_volatility_12h"] = np.sqrt(
        minute_return.pow(2).rolling(TWELVE_HOURS, min_periods=TWELVE_HOURS).sum()
    )

    absolute_price_change = frame["close"].diff().abs()
    direction = (frame["close"] - frame["close"].shift(TWENTY_FOUR_HOURS)).abs()
    path = absolute_price_change.rolling(
        TWENTY_FOUR_HOURS,
        min_periods=TWENTY_FOUR_HOURS,
    ).sum()
    kaufman_efficiency = direction / path
    features["ada_kaufman_efficiency_24h"] = kaufman_efficiency.mask(path.eq(0), 0.0)

    rolling_open = frame["open"].shift(TWELVE_HOURS - 1)
    rolling_high = frame["high"].rolling(TWELVE_HOURS, min_periods=TWELVE_HOURS).max()
    rolling_low = frame["low"].rolling(TWELVE_HOURS, min_periods=TWELVE_HOURS).min()
    rolling_range = rolling_high - rolling_low
    close_position = (frame["close"] - rolling_low) / rolling_range
    close_position = close_position.mask(rolling_range.eq(0), 0.5)
    close_position = close_position.mask(rolling_open.isna())
    features["ada_close_position_12h"] = close_position

    result = features.reset_index()[["timestamp", *HMM_FEATURE_COLUMNS]]
    if dropna:
        result = result.dropna(axis=0, how="any").reset_index(drop=True)
    for column in HMM_FEATURE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float32")
    _validate_hmm_dataset(result, allow_nan=not dropna)
    return result


def _validate_hmm_dataset(dataset: pd.DataFrame, *, allow_nan: bool = False) -> None:
    expected_columns = ["timestamp", *HMM_FEATURE_COLUMNS]
    if dataset.columns.tolist() != expected_columns:
        raise ValueError("Dataset columns do not match the selected HMM schema")
    if dataset.empty:
        return

    timestamps = pd.DatetimeIndex(pd.to_datetime(dataset["timestamp"], utc=True))
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Dataset timestamps must be sorted")
    if timestamps.duplicated().any():
        raise ValueError("Dataset contains duplicate timestamps")

    values = dataset.loc[:, list(HMM_FEATURE_COLUMNS)].to_numpy(dtype="float64")
    finite_mask = np.isfinite(values)
    if allow_nan:
        finite_mask |= np.isnan(values)
    if not finite_mask.all():
        raise ValueError("Selected HMM dataset contains non-finite values")


def _read_s3_inputs(s3, bucket: str, keys: list[str]) -> pd.DataFrame:
    frames = []
    for key in keys:
        frame = load_s3_parquet(s3, bucket, key)
        if frame is None:
            raise FileNotFoundError(f"Input not found: s3://{bucket}/{key}")
        frames.append(frame)
    if not frames:
        raise ValueError("At least one S3 input key is required")
    return pd.concat(frames, ignore_index=True)


def _date_range_input_keys(
    *,
    symbol: str,
    interval: str,
    raw_prefix: str,
    start_date: str,
    end_date: str,
) -> list[str]:
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if start > end:
        raise ValueError("start-date must not be later than end-date")

    # Include previous-day context so the first requested day can have 24h windows.
    source_start = start - timedelta(days=1)
    return [
        kline_key(symbol, interval, day.strftime("%Y-%m-%d"), raw_prefix)
        for day in pd.date_range(source_start, end, freq="D")
    ]


def _filter_output_range(
    dataset: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if start_date is None or end_date is None or dataset.empty:
        return dataset

    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC") + timedelta(days=1)
    mask = dataset["timestamp"].ge(start) & dataset["timestamp"].lt(end)
    return dataset.loc[mask].reset_index(drop=True)


def _write_s3_parquet(s3, bucket: str, key: str, dataset: pd.DataFrame) -> int:
    buffer = io.BytesIO()
    dataset.to_parquet(buffer, index=False, engine="pyarrow", compression="zstd")
    payload = buffer.getvalue()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="application/vnd.apache.parquet",
        Metadata={
            "rows": str(len(dataset)),
            "features": str(len(HMM_FEATURE_COLUMNS)),
            "frequency": "1m",
            "dataset-kind": "ada_hmm_12h_24h",
        },
    )
    return len(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare minute ADA 12h/24h feature data for GaussianHMM"
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument(
        "--input-key",
        nargs="+",
        default=None,
        help="One or more S3 keys with ADAUSDT 1m kline parquet partitions",
    )
    parser.add_argument("--start-date", help="First UTC date to keep in the output")
    parser.add_argument("--end-date", help="Last UTC date to keep in the output")
    parser.add_argument("--symbol", default="ADAUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--raw-prefix", default="raw")
    parser.add_argument(
        "--output-key",
        default=DEFAULT_OUTPUT_KEY,
        help="Destination S3 key for the prepared parquet dataset",
    )
    parser.add_argument(
        "--keep-warmup",
        action="store_true",
        help="Keep rows with warmup NaNs instead of dropping incomplete feature rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_key is None and (args.start_date is None or args.end_date is None):
        raise ValueError("Provide either --input-key or both --start-date and --end-date")

    input_keys = args.input_key or _date_range_input_keys(
        symbol=args.symbol,
        interval=args.interval,
        raw_prefix=args.raw_prefix,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    s3 = make_s3_client()
    klines = _read_s3_inputs(s3, args.bucket, input_keys)
    dataset = build_ada_hmm_12h_24h_dataset(klines, dropna=not args.keep_warmup)
    dataset = _filter_output_range(
        dataset,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    size = _write_s3_parquet(s3, args.bucket, args.output_key, dataset)
    print(
        f"Uploaded s3://{args.bucket}/{args.output_key} rows={len(dataset):,} "
        f"features={len(HMM_FEATURE_COLUMNS)} frequency=1min size_mib={size / 1024**2:.2f}"
    )
    print(f"input_keys={len(input_keys)}")
    print("features=" + ",".join(HMM_FEATURE_COLUMNS))


if __name__ == "__main__":
    main()
