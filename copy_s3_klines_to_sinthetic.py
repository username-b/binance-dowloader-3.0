import argparse
import os
from typing import Iterator

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


def normalize_prefix(prefix: str) -> str:
    return prefix.strip("/")


def iter_object_keys(s3_client, bucket: str, prefix: str) -> Iterator[str]:
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith("/"):
                yield key


def copy_prefix(
    s3_client,
    bucket: str,
    source_prefix: str,
    destination_prefix: str,
    dry_run: bool,
) -> int:
    copied = 0

    for source_key in iter_object_keys(s3_client, bucket, source_prefix):
        relative_parts = source_key[len(source_prefix):].split("/")
        date_index = next(
            (
                index
                for index, part in enumerate(relative_parts)
                if part.startswith("date=")
            ),
            None,
        )
        if date_index is None:
            print(f"Skip without date partition: {source_key}")
            continue

        destination_key = f"{destination_prefix}{'/'.join(relative_parts[date_index:])}"

        print(f"{source_key} -> {destination_key}")

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
            "Copy ADAUSDT klines from raw/klines/symbol=ADAUSDT to "
            "raw/sinthetic_data/date=... inside the configured S3 bucket."
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file with storage.bucket and storage.prefix.",
    )
    parser.add_argument(
        "--symbol",
        default="ADAUSDT",
        help="Symbol partition to copy.",
    )
    parser.add_argument(
        "--source-dir",
        default="klines",
        help="Directory under storage.prefix to copy from.",
    )
    parser.add_argument(
        "--destination-dir",
        default="sinthetic_data",
        help="Directory under storage.prefix to copy to.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned copies, do not write to S3.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_config(args.config)

    bucket = cfg["storage"]["bucket"]
    raw_prefix = normalize_prefix(cfg["storage"].get("prefix", ""))
    source_prefix = (
        f"{raw_prefix}/{args.source_dir.strip('/')}/symbol={args.symbol.strip('/')}/"
    ).lstrip("/")
    destination_prefix = (
        f"{raw_prefix}/{args.destination_dir.strip('/')}/"
    ).lstrip("/")

    print(f"Bucket: s3://{bucket}")
    print(f"Source: {source_prefix}")
    print(f"Destination: {destination_prefix}")
    if args.dry_run:
        print("Mode: dry-run")

    s3_client = build_s3_client()

    try:
        copied = copy_prefix(
            s3_client=s3_client,
            bucket=bucket,
            source_prefix=source_prefix,
            destination_prefix=destination_prefix,
            dry_run=args.dry_run,
        )
    except ClientError as exc:
        raise SystemExit(f"S3 copy failed: {exc}") from exc

    if copied == 0:
        print("No objects found to copy.")
        return

    action = "Would copy" if args.dry_run else "Copied"
    print(f"{action} {copied} object(s).")


if __name__ == "__main__":
    main()
