"""Build a read-only inventory of parquet datasets in the configured S3 bucket."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import boto3
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv


DATE_PARTITION_RE = re.compile(r"/date=(\d{4}-\d{2}-\d{2})/")


def make_s3_client():
    load_dotenv(".env")
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("YC_ENDPOINT"),
        region_name=os.getenv("YC_REGION"),
        aws_access_key_id=os.getenv("YC_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("YC_SECRET_ACCESS_KEY"),
    )


def dataset_path(key: str) -> str:
    parts = key.split("/")
    date_index = next(
        (index for index, part in enumerate(parts) if part.startswith("date=")),
        None,
    )
    if date_index is not None:
        return "/".join(parts[:date_index])
    return "[bucket-root]" if len(parts) == 1 else "/".join(parts[:-1])


def inventory_bucket(bucket: str) -> list[dict]:
    s3 = make_s3_client()
    grouped = defaultdict(
        lambda: {
            "files": 0,
            "bytes": 0,
            "dates": set(),
            "sample_key": None,
            "last_modified": None,
        }
    )

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue

            record = grouped[dataset_path(key)]
            record["files"] += 1
            record["bytes"] += obj.get("Size", 0)
            record["sample_key"] = record["sample_key"] or key

            date_match = DATE_PARTITION_RE.search(f"/{key}")
            if date_match:
                record["dates"].add(date_match.group(1))

            modified = obj.get("LastModified")
            if modified and (
                record["last_modified"] is None
                or modified > record["last_modified"]
            ):
                record["last_modified"] = modified

    result = []
    for path, record in sorted(grouped.items()):
        dates = sorted(record["dates"])
        sample = s3.get_object(Bucket=bucket, Key=record["sample_key"])
        parquet = pq.ParquetFile(io.BytesIO(sample["Body"].read()))
        schema = [
            {"column": field.name, "type": str(field.type)}
            for field in parquet.schema_arrow
        ]
        result.append(
            {
                "dataset": path,
                "files": record["files"],
                "size_gib": round(record["bytes"] / 1024**3, 3),
                "partition_count": len(dates),
                "date_start": dates[0] if dates else None,
                "date_end": dates[-1] if dates else None,
                "sample_key": record["sample_key"],
                "sample_rows": parquet.metadata.num_rows,
                "last_modified": (
                    record["last_modified"].isoformat()
                    if record["last_modified"]
                    else None
                ),
                "schema": schema,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="binance-data-downloader")
    parser.add_argument("--output-dir", type=Path, default=Path("s3_catalog"))
    args = parser.parse_args()

    inventory = inventory_bucket(args.bucket)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "inventory.json"
    csv_path = args.output_dir / "inventory.csv"
    json_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    flat_rows = [{key: value for key, value in row.items() if key != "schema"} for row in inventory]
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    print(f"Datasets: {len(inventory)}")
    print(f"JSON: {json_path.resolve()}")
    print(f"CSV:  {csv_path.resolve()}")


if __name__ == "__main__":
    main()
