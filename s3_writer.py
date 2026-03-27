import io
import pandas as pd
import boto3
import os
from botocore.exceptions import ClientError


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
        interval: str,
        date: str,
    ) -> str:
        return (
            f"{self.prefix}/{source}/"
            f"symbol={symbol}/"
            f"interval={interval}/"
            f"date={date}/data.parquet"
        ).lstrip("/")

    # =========================
    # 🔥 EXISTS CHECK
    # =========================
    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def write_df(
        self,
        df: pd.DataFrame,
        source: str,
        symbol: str,
        interval: str,
        date: str,
    ) -> None:
        key = self._build_key(source, symbol, interval, date)

        # =========================
        # 🔥 SKIP IF EXISTS
        # =========================
        if self.exists(key):
            print(f"Skip exists: s3://{self.bucket}/{key}")
            return

        # =========================
        # parquet в память
        # =========================
        buffer = io.BytesIO()

        df.to_parquet(
            buffer,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )

        # =========================
        # upload
        # =========================
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer.getvalue(),
        )

        print(f"Uploaded: s3://{self.bucket}/{key}")