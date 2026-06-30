"""Build selected-feature datasets for each forecast target and upload to S3.

This script uses the manually approved feature list CSV and the compact full
feature dataset:

    s3://binance-data-downloader/features/compact/unified_dataset_full.parquet

It creates three root-level S3 folders:

    dataset_target_10/
    dataset_target_20/
    dataset_target_30/

Each folder receives:

    dataset.parquet
    train.parquet  # timestamp < 2025-02-01
    test.parquet   # timestamp >= 2025-02-01

The script is intentionally not executed by Codex unless explicitly requested.
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path

import pandas as pd

from build_price_feature_day import load_s3_parquet, make_s3_client


DEFAULT_BUCKET = "binance-data-downloader"
DEFAULT_SOURCE_KEY = "features/compact/unified_dataset_full.parquet"
DEFAULT_SPLIT_TIMESTAMP = "2025-02-01 00:00:00+00:00"

DEFAULT_SELECTION_CSV = "итоговый набор признаков.csv"

TIMESTAMP_COLUMN = "timestamp"
VARIABLE_COLUMN = "Обозначение переменной"

TARGET_SPECS = {
    "target_10": {
        "target_column": "target_log_return_10m",
        "selection_column": "Отбор для target_log_return_10m",
        "s3_prefix": "dataset_target_10",
    },
    "target_20": {
        "target_column": "target_log_return_20m",
        "selection_column": "Отбор для target_log_return_20m",
        "s3_prefix": "dataset_target_20",
    },
    "target_30": {
        "target_column": "target_log_return_30m",
        "selection_column": "Отбор для target_log_return_30m",
        "s3_prefix": "dataset_target_30",
    },
}


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger("selected_target_datasets")


def load_selection_table(csv_path: str) -> pd.DataFrame:
    csv = Path(csv_path)
    if not csv.exists():
        raise FileNotFoundError(
            f"Selection CSV not found relative to project folder: {csv_path}"
        )
    return pd.read_csv(csv, encoding="utf-8-sig")


def normalize_yes(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"да", "yes", "y", "true", "1", "+"}


def selected_features(selection: pd.DataFrame, selection_column: str) -> list[str]:
    required = {VARIABLE_COLUMN, selection_column}
    missing = required.difference(selection.columns)
    if missing:
        raise ValueError(f"Selection table is missing columns: {sorted(missing)}")

    mask = selection[selection_column].map(normalize_yes)
    features = (
        selection.loc[mask, VARIABLE_COLUMN]
        .dropna()
        .map(lambda value: str(value).strip())
        .loc[lambda series: series.ne("")]
        .drop_duplicates()
        .tolist()
    )
    if not features:
        raise ValueError(f"No selected features found for {selection_column}")
    return features


def validate_columns(dataset: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in dataset.columns]
    if missing:
        raise ValueError(f"{label} columns are missing from full dataset: {missing}")


def build_target_dataset(
    full_dataset: pd.DataFrame,
    features: list[str],
    target_column: str,
) -> pd.DataFrame:
    columns = [TIMESTAMP_COLUMN, *features, target_column]
    validate_columns(full_dataset, columns, target_column)

    dataset = full_dataset.loc[:, columns].copy()
    dataset[TIMESTAMP_COLUMN] = pd.to_datetime(
        dataset[TIMESTAMP_COLUMN],
        utc=True,
        errors="raise",
    )
    dataset = dataset.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    return dataset


def split_train_test(
    dataset: pd.DataFrame,
    split_timestamp: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = pd.Timestamp(split_timestamp)
    if split.tzinfo is None:
        split = split.tz_localize("UTC")
    else:
        split = split.tz_convert("UTC")

    train = dataset.loc[dataset[TIMESTAMP_COLUMN] < split].reset_index(drop=True)
    test = dataset.loc[dataset[TIMESTAMP_COLUMN] >= split].reset_index(drop=True)
    return train, test


def upload_parquet(s3, bucket: str, key: str, dataset: pd.DataFrame) -> int:
    buffer = io.BytesIO()
    dataset.to_parquet(buffer, index=False, engine="pyarrow", compression="zstd")
    payload = buffer.getvalue()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="application/vnd.apache.parquet",
        Metadata={
            "rows": str(len(dataset)),
            "columns": str(len(dataset.columns)),
        },
    )
    return len(payload)


def upload_target_outputs(
    s3,
    bucket: str,
    prefix: str,
    dataset: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    outputs = {
        "dataset.parquet": dataset,
        "train.parquet": train,
        "test.parquet": test,
    }
    for filename, frame in outputs.items():
        key = f"{prefix.strip('/')}/{filename}"
        size = upload_parquet(s3, bucket, key, frame)
        logger.info(
            "uploaded s3://%s/%s rows=%d columns=%d size_mib=%.2f",
            bucket,
            key,
            len(frame),
            len(frame.columns),
            size / 1024**2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build selected-feature train/test datasets for target_10/20/30"
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--source-key", default=DEFAULT_SOURCE_KEY)
    parser.add_argument("--selection-csv", default=DEFAULT_SELECTION_CSV)
    parser.add_argument("--split-timestamp", default=DEFAULT_SPLIT_TIMESTAMP)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build datasets and print sizes without uploading to S3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    selection = load_selection_table(args.selection_csv)
    logger.info("loaded selection table rows=%d columns=%d", len(selection), len(selection.columns))

    logger.info("creating S3 client")
    s3 = make_s3_client()
    logger.info("S3 client ready")
    logger.info("loading source dataset s3://%s/%s", args.bucket, args.source_key)
    full_dataset = load_s3_parquet(s3, args.bucket, args.source_key)
    if full_dataset is None:
        raise FileNotFoundError(f"Source dataset not found: s3://{args.bucket}/{args.source_key}")
    logger.info(
        "loaded source dataset rows=%d columns=%d from s3://%s/%s",
        len(full_dataset),
        len(full_dataset.columns),
        args.bucket,
        args.source_key,
    )

    for target_name, spec in TARGET_SPECS.items():
        features = selected_features(selection, spec["selection_column"])
        target_dataset = build_target_dataset(
            full_dataset,
            features,
            spec["target_column"],
        )
        train, test = split_train_test(target_dataset, args.split_timestamp)
        logger.info(
            "%s target=%s features=%d full_rows=%d train_rows=%d test_rows=%d",
            target_name,
            spec["target_column"],
            len(features),
            len(target_dataset),
            len(train),
            len(test),
        )

        if args.dry_run:
            continue

        upload_target_outputs(
            s3,
            args.bucket,
            spec["s3_prefix"],
            target_dataset,
            train,
            test,
            logger,
        )


if __name__ == "__main__":
    main()
