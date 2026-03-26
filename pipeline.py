import datetime as dt
from typing import Optional

from binance_client import BinanceClient
from normalizer import KlinesNormalizer
from s3_writer import S3Writer


class Pipeline:
    def __init__(
        self,
        client: BinanceClient,
        normalizer: KlinesNormalizer,
        writer: S3Writer,
    ):
        self.client = client
        self.normalizer = normalizer
        self.writer = writer

    def run_range(
        self,
        source: str,
        symbol: str,
        interval: str,
        start_date: dt.date,
        end_date: dt.date,
    ):
        current = start_date

        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")

            try:
                print(f"Processing {date_str}...")

                df = self.client.load_day(
                    source=source,
                    symbol=symbol,
                    interval=interval,
                    date_str=date_str,
                )

                if df is None or df.empty:
                    print(f"Skip (no data): {date_str}")
                    current += dt.timedelta(days=1)
                    continue

                df = self.normalizer.normalize(df)

                self.writer.write_df(
                    df=df,
                    source=source,
                    symbol=symbol,
                    interval=interval,
                    date=date_str,
                )

                print(f"Done: {date_str}")

            except Exception as e:
                print(f"ERROR {date_str}: {e}")

            current += dt.timedelta(days=1)