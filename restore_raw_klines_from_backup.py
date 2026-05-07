import argparse
import os
from typing import Iterator, Sequence

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from config_loader import load_config


def build_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("YC_ENDPOINT"),
        region_name=os.getenv("YC_REGION"),
        aws_access_key_id=os.getenv("YC_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("YC_SECRET_ACCESS_KEY"),
    )


def iter_object_keys(s3_client, bucket: str, prefix: str) -> Iterator[str]:
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith("/"):
                yield key


def delete_keys(
    s3_client,
    bucket: str,
    keys: Sequence[str],
    dry_run: bool,
) -> int:
    deleted = 0

    for start in range(0, len(keys), 1000):
        chunk = keys[start:start + 1000]
        for key in chunk:
            print(f"DELETE {key}")

        if not dry_run:
            s3_client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": key} for key in chunk]},
            )

        deleted += len(chunk)

    return deleted


def copy_prefix(
    s3_client,
    bucket: str,
    source_prefix: str,
    destination_prefix: str,
    dry_run: bool,
) -> int:
    copied = 0

    for source_key in iter_object_keys(s3_client, bucket, source_prefix):
        relative_key = source_key[len(source_prefix):]
        destination_key = f"{destination_prefix}{relative_key}"

        print(f"COPY {source_key} -> {destination_key}")

        if not dry_run:
            s3_client.copy_object(
                Bucket=bucket,
                Key=destination_key,
                CopySource={"Bucket": bucket, "Key": source_key},
            )

        copied += 1

    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace raw/klines/symbol=ADAUSDT with objects from "
            "raw_backup/klines/symbol=ADAUSDT in the configured S3 bucket."
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file with storage.bucket.",
    )
    parser.add_argument(
        "--symbol",
        default="ADAUSDT",
        help="Symbol partition to restore.",
    )
    parser.add_argument(
        "--source-root",
        default="raw_backup",
        help="Backup root prefix to copy from.",
    )
    parser.add_argument(
        "--destination-root",
        default="raw",
        help="Destination root prefix to replace.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned deletes and copies, do not write to S3.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_config(args.config)

    bucket = cfg["storage"]["bucket"]
    symbol = args.symbol.strip("/")
    source_prefix = f"{args.source_root.strip('/')}/klines/symbol={symbol}/"
    destination_prefix = (
        f"{args.destination_root.strip('/')}/klines/symbol={symbol}/"
    )

    print(f"Bucket: s3://{bucket}")
    print(f"Source: {source_prefix}")
    print(f"Destination: {destination_prefix}")
    if args.dry_run:
        print("Mode: dry-run")

    s3_client = build_s3_client()

    try:
        source_keys = list(iter_object_keys(s3_client, bucket, source_prefix))
        if not source_keys:
            raise SystemExit("No backup objects found. Destination was not changed.")

        destination_keys = list(iter_object_keys(s3_client, bucket, destination_prefix))
        deleted = delete_keys(
            s3_client=s3_client,
            bucket=bucket,
            keys=destination_keys,
            dry_run=args.dry_run,
        )
        copied = copy_prefix(
            s3_client=s3_client,
            bucket=bucket,
            source_prefix=source_prefix,
            destination_prefix=destination_prefix,
            dry_run=args.dry_run,
        )
    except ClientError as exc:
        raise SystemExit(f"S3 restore failed: {exc}") from exc

    action = "Would replace" if args.dry_run else "Replaced"
    print(f"{action} destination: deleted {deleted}, copied {copied} object(s).")


if __name__ == "__main__":
    main()
