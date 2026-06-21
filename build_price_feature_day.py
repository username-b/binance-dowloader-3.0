"""Build one daily ADA price-feature partition from raw S3 klines."""

from __future__ import annotations

import argparse
import gc
import io
import os
from datetime import timedelta

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from price_features import FEATURE_COLUMNS, TARGET_COLUMNS, build_price_features


DEFAULT_START_DATE = "2020-02-01"
DEFAULT_END_DATE = "2026-02-01"
DEFAULT_OUTPUT_PREFIX = "features/unified_dataset"
DATASET_SCHEMA_VERSION = "2"
TRADE_SOURCE = "trades"


def make_s3_client():
    load_dotenv()
    required = (
        "YC_ENDPOINT",
        "YC_REGION",
        "YC_ACCESS_KEY_ID",
        "YC_SECRET_ACCESS_KEY",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing S3 environment variables: {', '.join(missing)}")

    return boto3.client(
        "s3",
        endpoint_url=os.getenv("YC_ENDPOINT"),
        region_name=os.getenv("YC_REGION"),
        aws_access_key_id=os.getenv("YC_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("YC_SECRET_ACCESS_KEY"),
    )


def kline_key(
    symbol: str,
    interval: str,
    date: str,
    raw_prefix: str = "raw",
) -> str:
    return (
        f"{raw_prefix.strip('/')}/klines/symbol={symbol}/"
        f"interval={interval}/date={date}/data.parquet"
    )


def trades_key(
    symbol: str,
    date: str,
    raw_prefix: str = "raw",
) -> str:
    return f"{raw_prefix.strip('/')}/trades/symbol={symbol}/date={date}/data.parquet"


def unified_dataset_key(
    symbol: str,
    interval: str,
    date: str,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
) -> str:
    return (
        f"{output_prefix.strip('/')}/symbol={symbol}/interval={interval}/"
        f"date={date}/data.parquet"
    )


def s3_partition_is_current(s3, bucket: str, key: str) -> bool:
    try:
        response = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    metadata = {name.lower(): value for name, value in response.get("Metadata", {}).items()}
    return (
        metadata.get("schema-version") == DATASET_SCHEMA_VERSION
        and metadata.get("trade-source") == TRADE_SOURCE
    )


def load_s3_parquet(s3, bucket: str, key: str) -> pd.DataFrame | None:
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    return pd.read_parquet(io.BytesIO(body))


def build_day_dataset(
    s3,
    bucket: str,
    symbol: str,
    interval: str,
    date: str,
    raw_prefix: str = "raw",
    btc_symbol: str = "BTCUSDT",
    require_previous_context: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    day = pd.Timestamp(date, tz="UTC")
    previous_day = (day - timedelta(days=1)).strftime("%Y-%m-%d")
    current_day = day.strftime("%Y-%m-%d")
    next_day = (day + timedelta(days=1)).strftime("%Y-%m-%d")

    loaded_frames = []
    loaded_trades = []
    loaded_btc_frames = []
    loaded_keys = []
    for source_date in (previous_day, current_day):
        key = kline_key(symbol, interval, source_date, raw_prefix)
        frame = load_s3_parquet(s3, bucket, key)
        if frame is not None:
            loaded_frames.append(frame)
            loaded_keys.append(key)

        source_trades_key = trades_key(symbol, source_date, raw_prefix)
        trades = load_s3_parquet(s3, bucket, source_trades_key)
        if trades is not None:
            loaded_trades.append(trades)
            loaded_keys.append(source_trades_key)

        btc_key = kline_key(btc_symbol, interval, source_date, raw_prefix)
        btc_frame = load_s3_parquet(s3, bucket, btc_key)
        if btc_frame is not None:
            loaded_btc_frames.append(btc_frame)
            loaded_keys.append(btc_key)

    current_kline_key = kline_key(symbol, interval, current_day, raw_prefix)
    current_trades_key = trades_key(symbol, current_day, raw_prefix)
    current_btc_key = kline_key(btc_symbol, interval, current_day, raw_prefix)
    if require_previous_context:
        required_previous_keys = (
            kline_key(symbol, interval, previous_day, raw_prefix),
            trades_key(symbol, previous_day, raw_prefix),
            kline_key(btc_symbol, interval, previous_day, raw_prefix),
        )
        missing_previous = [key for key in required_previous_keys if key not in loaded_keys]
        if missing_previous:
            raise FileNotFoundError(
                "Previous-day context is incomplete: "
                + ", ".join(f"s3://{bucket}/{key}" for key in missing_previous)
            )
    if current_kline_key not in loaded_keys:
        key = kline_key(symbol, interval, current_day, raw_prefix)
        raise FileNotFoundError(f"Current-day source not found: s3://{bucket}/{key}")
    if current_trades_key not in loaded_keys:
        raise FileNotFoundError(
            f"Current-day source not found: s3://{bucket}/{current_trades_key}"
        )
    if current_btc_key not in loaded_keys:
        raise FileNotFoundError(f"Current-day source not found: s3://{bucket}/{current_btc_key}")

    next_kline_key = kline_key(symbol, interval, next_day, raw_prefix)
    future_klines = load_s3_parquet(s3, bucket, next_kline_key)
    if future_klines is None:
        raise FileNotFoundError(
            f"Next-day source required for forward targets: s3://{bucket}/{next_kline_key}"
        )
    loaded_keys.append(next_kline_key)

    context = pd.concat(loaded_frames, ignore_index=True)
    trades_context = pd.concat(loaded_trades, ignore_index=True)
    btc_context = pd.concat(loaded_btc_frames, ignore_index=True)
    feature_frame = build_price_features(
        context,
        trades=trades_context,
        btc_klines=btc_context,
        future_klines=future_klines,
    )
    end = day + timedelta(days=1)
    result = feature_frame.loc[
        feature_frame["timestamp"].ge(day) & feature_frame["timestamp"].lt(end)
    ].reset_index(drop=True)

    expected_index = pd.date_range(day, periods=1440, freq="1min", name="timestamp")
    result = result.set_index("timestamp").reindex(expected_index).reset_index()
    return result[["timestamp", *FEATURE_COLUMNS, *TARGET_COLUMNS]], loaded_keys


def validate_day_dataset(
    dataset: pd.DataFrame,
    date: str,
    history_start_date: str = DEFAULT_START_DATE,
) -> None:
    expected_columns = ["timestamp", *FEATURE_COLUMNS, *TARGET_COLUMNS]
    if dataset.columns.tolist() != expected_columns:
        raise ValueError("Dataset columns do not match the declared schema")
    if len(dataset) != 1440:
        raise ValueError(f"Expected 1440 rows, got {len(dataset)}")

    day = pd.Timestamp(date, tz="UTC")
    expected_index = pd.date_range(day, periods=1440, freq="1min")
    timestamps = pd.DatetimeIndex(dataset["timestamp"])
    if not timestamps.equals(expected_index):
        raise ValueError("Dataset does not contain one ordered row for every UTC minute")

    if not dataset[list(TARGET_COLUMNS)].notna().all().all():
        raise ValueError("Forward targets contain missing values")

    complete_feature_rows = dataset[list(FEATURE_COLUMNS)].notna().all(axis=1).sum()
    expected_complete_rows = 1350 if date == history_start_date else 1440
    if complete_feature_rows != expected_complete_rows:
        raise ValueError(
            f"Expected {expected_complete_rows} complete feature rows, "
            f"got {complete_feature_rows}"
        )

    numeric = dataset.select_dtypes(include=[np.number])
    values = numeric.to_numpy(dtype="float64")
    if not np.isfinite(values[~np.isnan(values)]).all():
        raise ValueError("Dataset contains infinite numeric values")


def upload_day_dataset(
    s3,
    bucket: str,
    key: str,
    dataset: pd.DataFrame,
) -> int:
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
            "features": str(len(FEATURE_COLUMNS)),
            "targets": str(len(TARGET_COLUMNS)),
            "schema-version": DATASET_SCHEMA_VERSION,
            "trade-source": TRADE_SOURCE,
        },
    )
    return len(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Build one date; overrides start/end dates")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--symbol", default="ADAUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--btc-symbol", default="BTCUSDT")
    parser.add_argument("--bucket", default=os.getenv("YC_BUCKET", "binance-data-downloader"))
    parser.add_argument("--raw-prefix", default="raw")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = args.date or args.start_date
    end_date = args.date or args.end_date
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if start > end:
        raise ValueError("start-date must not be later than end-date")

    s3 = make_s3_client()
    dates = pd.date_range(start, end, freq="D")
    uploaded = 0
    skipped = 0
    failed = []
    uploaded_bytes = 0

    print(f"Range: {start} -> {end} ({len(dates)} days)")
    print(f"Destination: s3://{args.bucket}/{args.output_prefix.strip('/')}/")
    for position, day in enumerate(dates, start=1):
        date = day.strftime("%Y-%m-%d")
        destination_key = unified_dataset_key(
            args.symbol,
            args.interval,
            date,
            args.output_prefix,
        )
        label = f"[{position}/{len(dates)}] {date}"

        if not args.overwrite and s3_partition_is_current(
            s3, args.bucket, destination_key
        ):
            skipped += 1
            print(f"{label} skip exists")
            continue

        try:
            dataset, loaded_keys = build_day_dataset(
                s3=s3,
                bucket=args.bucket,
                symbol=args.symbol,
                interval=args.interval,
                date=date,
                raw_prefix=args.raw_prefix,
                btc_symbol=args.btc_symbol,
                require_previous_context=date != DEFAULT_START_DATE,
            )
            validate_day_dataset(dataset, date)
            size = upload_day_dataset(
                s3,
                args.bucket,
                destination_key,
                dataset,
            )
            uploaded += 1
            uploaded_bytes += size
            print(
                f"{label} uploaded rows={len(dataset)} "
                f"size_mib={size / 1024**2:.2f} sources={len(loaded_keys)}"
            )
        except Exception as exc:
            failed.append((date, str(exc)))
            print(f"{label} ERROR: {exc}")
            if not args.continue_on_error:
                raise
        finally:
            if "dataset" in locals():
                del dataset
            gc.collect()

    print(
        f"Completed: uploaded={uploaded}, skipped={skipped}, failed={len(failed)}, "
        f"uploaded_gib={uploaded_bytes / 1024**3:.3f}"
    )
    if failed:
        for date, message in failed:
            print(f"FAILED {date}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
