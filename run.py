import argparse
import datetime as dt
import os

from dotenv import load_dotenv

from binance_client import BinanceClient
from normalizer import get_normalizer
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

    client = BinanceClient(
        base_root="https://data.binance.vision/data/futures/um/daily",
        max_retries=cfg["download"]["retries"],
        timeout=tuple(cfg["download"]["timeout"]),
    )

    normalizer = get_normalizer(cfg["source"])

    writer = S3Writer(
        bucket=cfg["storage"]["bucket"],
        prefix=cfg["storage"]["prefix"],
    )

    pipeline = Pipeline(client, normalizer, writer)

    start_date = cfg["date_range"]["start"]
    end_date = cfg["date_range"]["end"]
    interval = cfg.get("interval")

    for symbol in cfg["symbols"]:
        print(f"\n=== PROCESSING {symbol} ===\n")

        pipeline.run_range(
            source=cfg["source"],
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
        )


if __name__ == "__main__":
    main()
