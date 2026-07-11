import argparse
import datetime as dt
import io
import os
from typing import Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from binance_client import BinanceClient
from config_loader import load_config
from normalizer import KlinesNormalizer


BINANCE_FUTURES_DAILY_ROOT = "https://data.binance.vision/data/futures/um/daily"


def parse_date(value) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def iter_dates(start_date: dt.date, end_date: dt.date):
    current = start_date
    while current <= end_date:
        yield current
        current += dt.timedelta(days=1)


def make_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("YC_ENDPOINT"),
        region_name=os.getenv("YC_REGION"),
        aws_access_key_id=os.getenv("YC_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("YC_SECRET_ACCESS_KEY"),
    )


def s3_key_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def put_parquet(
    s3_client,
    bucket: str,
    key: str,
    df: pd.DataFrame,
    overwrite: bool = True,
) -> bool:
    if not overwrite and s3_key_exists(s3_client, bucket, key):
        print(f"Skip exists: s3://{bucket}/{key}")
        return False

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow", compression="zstd")
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"Uploaded: s3://{bucket}/{key} rows={len(df)}")
    return True


def raw_klines_key(prefix: str, symbol: str, date: str) -> str:
    return (
        f"{prefix.strip('/')}/raw/klines/"
        f"symbol={symbol}/interval=1m/date={date}/data.parquet"
    ).lstrip("/")


def resolve_returns_key(prefix: str, template: str, start_date: dt.date, end_date: dt.date) -> str:
    key = template.format(start=start_date.isoformat(), end=end_date.isoformat())
    return f"{prefix.strip('/')}/{key.lstrip('/')}".lstrip("/")


def build_returns_for_symbol(
    client: BinanceClient,
    normalizer: KlinesNormalizer,
    s3_client,
    bucket: str,
    prefix: str,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    skip_existing_raw: bool,
) -> Optional[pd.DataFrame]:
    previous_close = None
    frames = []

    for current in iter_dates(start_date, end_date):
        date_str = current.isoformat()
        print(f"[{symbol}] Processing {date_str}...")

        raw_df = client.load_day(
            source="klines",
            symbol=symbol,
            interval="1m",
            date_str=date_str,
        )

        if raw_df is None or raw_df.empty:
            print(f"[{symbol}] Skip no source data: {date_str}")
            continue

        df = normalizer.normalize(raw_df)
        if df.empty:
            print(f"[{symbol}] Skip empty after normalize: {date_str}")
            continue

        key = raw_klines_key(prefix=prefix, symbol=symbol, date=date_str)
        put_parquet(
            s3_client=s3_client,
            bucket=bucket,
            key=key,
            df=df,
            overwrite=not skip_existing_raw,
        )

        close = df["close"].astype("float64")
        returns = close.pct_change()
        if previous_close is not None and len(close) > 0:
            returns.iloc[0] = close.iloc[0] / previous_close - 1.0
        previous_close = close.iloc[-1]

        day_returns = pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                f"{symbol}_return": returns.astype("float32"),
            }
        )
        frames.append(day_returns)

    if not frames:
        return None

    symbol_returns = pd.concat(frames, ignore_index=True)
    symbol_returns = symbol_returns.drop_duplicates(subset=["timestamp"], keep="last")
    symbol_returns = symbol_returns.sort_values("timestamp").reset_index(drop=True)
    return symbol_returns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    cfg = load_config(args.config)

    symbols = cfg["symbols"]
    start_date = parse_date(cfg["date_range"]["start"])
    end_date = parse_date(cfg["date_range"]["end"])
    storage_cfg = cfg["storage"]
    download_cfg = cfg.get("download", {})
    output_cfg = cfg.get("output", {})

    bucket = storage_cfg["bucket"]
    prefix = storage_cfg.get("prefix", "minute_returns_dataset")
    skip_existing_raw = output_cfg.get("skip_existing_raw", True)
    overwrite_returns = output_cfg.get("overwrite_returns", True)
    returns_template = output_cfg.get(
        "returns_key",
        "features/minute_returns_1m_{start}_{end}.parquet",
    )

    s3_client = make_s3_client()
    client = BinanceClient(
        base_root=BINANCE_FUTURES_DAILY_ROOT,
        max_retries=download_cfg.get("retries", 3),
        timeout=tuple(download_cfg.get("timeout", [10, 60])),
    )
    normalizer = KlinesNormalizer()

    returns_by_symbol = []
    for symbol in symbols:
        symbol_returns = build_returns_for_symbol(
            client=client,
            normalizer=normalizer,
            s3_client=s3_client,
            bucket=bucket,
            prefix=prefix,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            skip_existing_raw=skip_existing_raw,
        )

        if symbol_returns is None:
            print(f"[{symbol}] No returns collected")
            continue

        returns_by_symbol.append(symbol_returns.set_index("timestamp"))

    if not returns_by_symbol:
        raise RuntimeError("No data collected for any configured symbol")

    wide_returns = pd.concat(returns_by_symbol, axis=1, join="outer")
    wide_returns = wide_returns.sort_index().reset_index()
    returns_key = resolve_returns_key(
        prefix=prefix,
        template=returns_template,
        start_date=start_date,
        end_date=end_date,
    )
    put_parquet(
        s3_client=s3_client,
        bucket=bucket,
        key=returns_key,
        df=wide_returns,
        overwrite=overwrite_returns,
    )

    print(f"Done. Returns columns: {', '.join(wide_returns.columns)}")


if __name__ == "__main__":
    main()
