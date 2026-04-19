import io
import pandas as pd
import boto3
import os
from typing import Optional


class S3Writer:
    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix.strip("/")

        self.s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("YC_ENDPOINT"),
            region_name=os.getenv("YC_REGION"),
            aws_access_key_id=os.getenv("YC_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("YC_SECRET_ACCESS_KEY"),
        )

    def _build_key(
        self,
        source: str,
        symbol: str,
        interval: Optional[str],
        date: str,
    ) -> str:
        parts = [
            self.prefix,
            source,
            f"symbol={symbol}",
        ]

        if interval:
            parts.append(f"interval={interval}")

        parts.extend(
            [
                f"date={date}",
                "data.parquet",
            ]
        )

        return "/".join(part for part in parts if part).lstrip("/")

    def write_df(
        self,
        df: pd.DataFrame,
        source: str,
        symbol: str,
        interval: Optional[str],
        date: str,
    ) -> None:
        # =========================
        # 1. parquet в память
        # =========================
        buffer = io.BytesIO()

        df.to_parquet(
            buffer,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )

        # =========================
        # 2. ключ
        # =========================
        key = self._build_key(source, symbol, interval, date)

        # =========================
        # 3. upload
        # =========================
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer.getvalue(),
        )

        print(f"Uploaded: s3://{self.bucket}/{key}")
