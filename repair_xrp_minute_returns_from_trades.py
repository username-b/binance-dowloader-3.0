"""Repair missing XRPUSDT minute returns from Binance raw trades.

The script downloads non-aggregated USD-M futures trades for the affected XRPUSDT
dates, aggregates them to minute closes, computes minute returns, fills the
cached wide minute-returns dataset, and optionally uploads the repaired parquet
to S3.
"""

from __future__ import annotations

import argparse
import io
import os
import time
import zipfile
from pathlib import Path
from typing import Iterable

import boto3
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


BASE_ROOT = "https://data.binance.vision/data/futures/um/daily"
BUCKET = "binance-data-downloader"
INPUT_KEY = "minute_returns_dataset/features/minute_returns_1m_2020-02-01_2026-02-01.parquet"
OUTPUT_KEY = (
    "minute_returns_dataset/features/"
    "minute_returns_1m_2020-02-01_2026-02-01_xrp_repaired.parquet"
)
DEFAULT_LOCAL_INPUT = Path("analysis/cache/minute_returns_1m_2020-02-01_2026-02-01.parquet")
DEFAULT_LOCAL_OUTPUT = Path(
    "analysis/cache/minute_returns_1m_2020-02-01_2026-02-01_xrp_repaired.parquet"
)
DEFAULT_TRADES_CACHE = Path("analysis/cache/xrpusdt_trades")
SYMBOL = "XRPUSDT"
RETURN_COLUMN = f"{SYMBOL}_return"
TRADE_DATES = [
    "2020-11-29",
    "2020-11-30",
    "2020-12-01",
    "2020-12-02",
    "2020-12-03",
    "2020-12-04",
    "2020-12-05",
]
REPAIR_START = pd.Timestamp("2020-11-30 00:00:00", tz="UTC")
REPAIR_END_EXCLUSIVE = pd.Timestamp("2020-12-06 00:00:00", tz="UTC")

TRADE_COLUMNS = [
    "trade_id",
    "price",
    "qty",
    "quote_qty",
    "time",
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


def trade_url(symbol: str, date: str) -> str:
    return f"{BASE_ROOT}/trades/{symbol}/{symbol}-trades-{date}.zip"


def download_zip(url: str, retries: int = 3, timeout: tuple[int, int] = (10, 300)) -> bytes | None:
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return response.content
            if response.status_code == 404:
                return None
            if attempt == retries:
                response.raise_for_status()
        except requests.RequestException:
            if attempt == retries:
                raise
        time.sleep(2**attempt)
    return None


def read_csv_from_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as source:
            sample = source.read(512)
            source.seek(0)
            has_header = sample.lower().startswith(b"id,") or sample.lower().startswith(b"trade_id,")
            return pd.read_csv(
                source,
                header=0 if has_header else None,
                names=None if has_header else TRADE_COLUMNS,
                low_memory=False,
            )


def normalize_trades(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.iloc[:, : len(TRADE_COLUMNS)].copy()
    frame.columns = TRADE_COLUMNS
    frame["trade_id"] = pd.to_numeric(frame["trade_id"], errors="coerce").astype("Int64")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["qty"] = pd.to_numeric(frame["qty"], errors="coerce")
    frame["quote_qty"] = pd.to_numeric(frame["quote_qty"], errors="coerce")
    frame["time"] = pd.to_numeric(frame["time"], errors="coerce").astype("Int64")
    frame["is_buyer_maker"] = frame["is_buyer_maker"].astype("boolean")
    frame = frame.dropna(subset=["trade_id", "price", "qty", "quote_qty", "time"])
    frame["trade_id"] = frame["trade_id"].astype("int64")
    frame["time"] = frame["time"].astype("int64")
    frame["timestamp"] = pd.to_datetime(frame["time"], unit="ms", utc=True)
    return (
        frame.drop_duplicates("trade_id")
        .sort_values(["time", "trade_id"])
        .reset_index(drop=True)
    )


def load_or_download_trades(
    dates: Iterable[str],
    cache_dir: Path,
    force_download: bool = False,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for date in dates:
        parquet_path = cache_dir / f"{SYMBOL}-trades-{date}.parquet"
        if parquet_path.exists() and not force_download:
            trades = pd.read_parquet(parquet_path)
            print(f"Using cached trades {parquet_path}: rows={len(trades):,}")
            frames.append(trades)
            continue

        url = trade_url(SYMBOL, date)
        print(f"Downloading {url}")
        content = download_zip(url)
        if content is None:
            raise FileNotFoundError(f"No Binance trades file: {url}")
        trades = normalize_trades(read_csv_from_zip(content))
        trades.to_parquet(parquet_path, index=False, compression="zstd")
        print(f"Saved {parquet_path}: rows={len(trades):,}")
        frames.append(trades)
    return pd.concat(frames, ignore_index=True).sort_values(["time", "trade_id"])


def minute_returns_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.sort_values(["timestamp", "trade_id"]).copy()
    trades["minute"] = trades["timestamp"].dt.floor("min")
    minute = (
        trades.groupby("minute", as_index=False)
        .agg(close=("price", "last"), trades=("trade_id", "size"))
        .rename(columns={"minute": "timestamp"})
    )
    full_index = pd.date_range(
        minute["timestamp"].min().floor("D"),
        minute["timestamp"].max().ceil("D") - pd.Timedelta(minutes=1),
        freq="1min",
        tz="UTC",
    )
    minute = minute.set_index("timestamp").reindex(full_index)
    minute.index.name = "timestamp"
    minute["close"] = minute["close"].ffill()
    minute[RETURN_COLUMN] = minute["close"].pct_change()
    return minute.reset_index()[["timestamp", RETURN_COLUMN, "close", "trades"]]


def repair_dataset(
    input_path: Path,
    output_path: Path,
    repaired_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = pd.read_parquet(input_path)
    original["timestamp"] = pd.to_datetime(original["timestamp"], utc=True)
    original = original.sort_values("timestamp").drop_duplicates("timestamp")

    full_index = pd.date_range(
        original["timestamp"].min(),
        original["timestamp"].max(),
        freq="1min",
        tz="UTC",
    )
    repaired = original.set_index("timestamp").reindex(full_index)
    repaired.index.name = "timestamp"

    fill_values = repaired_returns.set_index("timestamp")[RETURN_COLUMN].reindex(repaired.index)
    target = repaired[RETURN_COLUMN]
    repair_window = (repaired.index >= REPAIR_START) & (repaired.index < REPAIR_END_EXCLUSIVE)
    repair_mask = repair_window & fill_values.notna()
    before = target.loc[repair_mask].copy()
    repaired.loc[repair_mask, RETURN_COLUMN] = fill_values.loc[repair_mask].astype(
        target.dtype,
        copy=False,
    )

    repaired = repaired.reset_index()
    all_nan_rows = repaired.drop(columns=["timestamp"]).isna().all(axis=1)
    if all_nan_rows.any():
        repaired = repaired.loc[~all_nan_rows].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_parquet(output_path, index=False, compression="zstd")

    changed_mask = before.isna() | ~np.isclose(
        before.astype("float64"),
        fill_values.loc[repair_mask].astype("float64"),
        rtol=0.0,
        atol=1e-12,
        equal_nan=False,
    )
    changed_index = before.index[changed_mask]
    changed = pd.DataFrame(
        {
            "timestamp": changed_index,
            "symbol": SYMBOL,
            "old_return": before.loc[changed_index].to_numpy(),
            "repaired_return": fill_values.loc[changed_index].to_numpy(),
        }
    )
    changed["action"] = np.where(changed["old_return"].isna(), "filled", "overwritten")
    return repaired, changed[["timestamp", "symbol", "action", "old_return", "repaired_return"]]


def compare_with_existing(input_path: Path, repaired_returns: pd.DataFrame) -> pd.DataFrame:
    original = pd.read_parquet(input_path, columns=["timestamp", RETURN_COLUMN])
    original["timestamp"] = pd.to_datetime(original["timestamp"], utc=True)
    check = original.merge(
        repaired_returns[["timestamp", RETURN_COLUMN]].rename(
            columns={RETURN_COLUMN: "trade_return"}
        ),
        on="timestamp",
        how="inner",
    )
    check = check[check[RETURN_COLUMN].notna() & check["trade_return"].notna()].copy()
    check["abs_diff"] = (check[RETURN_COLUMN] - check["trade_return"]).abs()
    by_date = (
        check.groupby(check["timestamp"].dt.date)
        .agg(
            compared=("abs_diff", "size"),
            max_abs_diff=("abs_diff", "max"),
            mean_abs_diff=("abs_diff", "mean"),
        )
        .reset_index(names="date")
    )
    return by_date


def upload_file(local_path: Path, bucket: str, key: str) -> None:
    s3 = make_s3_client()
    s3.upload_file(
        str(local_path),
        bucket,
        key,
        ExtraArgs={"ContentType": "application/vnd.apache.parquet"},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_LOCAL_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_LOCAL_OUTPUT)
    parser.add_argument("--trades-cache", type=Path, default=DEFAULT_TRADES_CACHE)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--output-key", default=OUTPUT_KEY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trades = load_or_download_trades(
        TRADE_DATES,
        cache_dir=args.trades_cache,
        force_download=args.force_download,
    )
    repaired_returns = minute_returns_from_trades(trades)
    comparison = compare_with_existing(args.input, repaired_returns)
    print("Comparison on already-present XRP returns:")
    print(comparison.to_string(index=False))

    repaired, changed = repair_dataset(args.input, args.output, repaired_returns)
    changed_path = args.output.with_name(args.output.stem + "_changed_rows.csv")
    changed.to_csv(changed_path, index=False)
    print("Changed XRP returns:")
    if changed.empty:
        print("no changes")
    else:
        print(changed.groupby(["action", changed["timestamp"].dt.date]).size().to_string())
    print(f"Saved repaired dataset: {args.output} rows={len(repaired):,}")
    print(f"Saved changed-row manifest: {changed_path}")

    if args.upload:
        upload_file(args.output, args.bucket, args.output_key)
        print(f"Uploaded s3://{args.bucket}/{args.output_key}")


if __name__ == "__main__":
    main()
