import datetime as dt
import io
import os
import re

import boto3
import pandas as pd
from dotenv import load_dotenv

from binance_client import BinanceClient


AGGTRADES_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]


def make_s3_client():
    load_dotenv()
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("YC_ENDPOINT"),
        region_name=os.getenv("YC_REGION"),
        aws_access_key_id=os.getenv("YC_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("YC_SECRET_ACCESS_KEY"),
    )


def aggtrades_key(symbol: str, date: str, raw_prefix: str = "raw") -> str:
    return f"{raw_prefix.strip('/')}/aggTrades/symbol={symbol}/date={date}/data.parquet"


def s3_key_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        error_code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def normalize_aggtrades(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] < len(AGGTRADES_COLUMNS):
        raise ValueError(
            f"aggTrades source has {df.shape[1]} columns, expected at least {len(AGGTRADES_COLUMNS)}"
        )

    normalized = df.iloc[:, : len(AGGTRADES_COLUMNS)].copy()
    normalized.columns = AGGTRADES_COLUMNS

    float_columns = ["price", "quantity"]
    int_columns = [
        "agg_trade_id",
        "first_trade_id",
        "last_trade_id",
        "transact_time",
    ]

    for column in float_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("float32")

    for column in int_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("Int64")

    normalized["is_buyer_maker"] = normalized["is_buyer_maker"].astype("boolean")
    normalized = normalized.dropna(subset=["transact_time", "price", "quantity", "is_buyer_maker"])

    for column in int_columns:
        normalized[column] = normalized[column].astype("int64")

    normalized["timestamp"] = pd.to_datetime(normalized["transact_time"], unit="ms", utc=True)
    normalized = (
        normalized.drop_duplicates(subset=["agg_trade_id"])
        .sort_values(["transact_time", "agg_trade_id"])
        .reset_index(drop=True)
    )

    return normalized[
        [
            "timestamp",
            "transact_time",
            "agg_trade_id",
            "first_trade_id",
            "last_trade_id",
            "price",
            "quantity",
            "is_buyer_maker",
        ]
    ]


def expected_dates(start_date, end_date) -> list[str]:
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    return [
        day.strftime("%Y-%m-%d")
        for day in pd.date_range(start=start, end=end, freq="D")
    ]


def find_missing_aggtrades_dates(
    symbol: str,
    start_date,
    end_date,
    bucket: str,
    raw_prefix: str = "raw",
    s3_client=None,
) -> list[str]:
    s3 = s3_client or make_s3_client()
    existing_dates = set()
    source_prefix = f"{raw_prefix.strip('/')}/aggTrades/symbol={symbol}/"
    pattern = re.compile(r"/date=(\d{4}-\d{2}-\d{2})/data\.parquet$")

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=source_prefix):
        for obj in page.get("Contents", []):
            match = pattern.search(f"/{obj['Key']}")
            if match:
                existing_dates.add(match.group(1))

    return [date for date in expected_dates(start_date, end_date) if date not in existing_dates]


def backfill_missing_aggtrades_dates(
    symbol: str,
    start_date,
    end_date,
    bucket: str,
    raw_prefix: str = "raw",
    retries: int = 3,
    timeout: tuple[int, int] = (10, 60),
    base_root: str = "https://data.binance.vision/data/futures/um/daily",
    s3_client=None,
) -> pd.DataFrame:
    s3 = s3_client or make_s3_client()
    missing_dates = find_missing_aggtrades_dates(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        bucket=bucket,
        raw_prefix=raw_prefix,
        s3_client=s3,
    )

    client = BinanceClient(
        base_root=base_root,
        max_retries=retries,
        timeout=timeout,
    )

    rows = []
    for date in missing_dates:
        key = aggtrades_key(symbol=symbol, date=date, raw_prefix=raw_prefix)
        raw_df = client.load_day(source="aggTrades", symbol=symbol, date_str=date)
        if raw_df is None or raw_df.empty:
            rows.append({"symbol": symbol, "date": date, "rows": 0, "key": key, "status": "no_source_data"})
            continue

        normalized = normalize_aggtrades(raw_df)
        buffer = io.BytesIO()
        normalized.to_parquet(buffer, index=False, engine="pyarrow", compression="zstd")
        s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        rows.append({"symbol": symbol, "date": date, "rows": len(normalized), "key": key, "status": "uploaded"})

    return pd.DataFrame(rows, columns=["symbol", "date", "rows", "key", "status"])
