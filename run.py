import argparse
import datetime as dt
import os

from dotenv import load_dotenv

from binance_client import BinanceClient
from normalizer import KlinesNormalizer
from s3_writer import S3Writer
from pipeline import Pipeline
from config_loader import load_config

def parse_date(x):
    if isinstance(x, str):
        return dt.datetime.strptime(x, "%Y-%m-%d").date()
    return x    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    load_dotenv()

    args = parse_args()
    cfg = load_config(args.config)
    sharding_cfg = cfg.get("sharding", {})
    shard_id = sharding_cfg.get("shard_id", 0)
    shard_total = sharding_cfg.get("shard_total", 1)

    client = BinanceClient(
        base_root="https://data.binance.vision/data/futures/um/daily",
        max_retries=cfg["download"]["retries"],
        timeout=tuple(cfg["download"]["timeout"]),
    )

    normalizer = KlinesNormalizer()

    writer = S3Writer(
        bucket=cfg["storage"]["bucket"],
        prefix=cfg["storage"]["prefix"],
    )

    pipeline = Pipeline(client, normalizer, writer)

    start_date = cfg["date_range"]["start"]
    end_date = cfg["date_range"]["end"]

    sharding_cfg = cfg.get("sharding", {})

    shard_id = sharding_cfg.get("shard_id", 0)
    shard_total = sharding_cfg.get("shard_total", 1)

    print(f"Sharding: {shard_id}/{shard_total}")

    for symbol in cfg["symbols"]:
        print(f"\n=== PROCESSING {symbol} ===\n")

        pipeline.run_range(
            source=cfg["source"],
            symbol=symbol,
            interval=cfg["interval"],
            start_date=start_date,
            end_date=end_date,
            shard_id=shard_id,
            shard_total=shard_total,
        )


if __name__ == "__main__":
    main()