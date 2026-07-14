"""Add ADA 12h/24h HMM posterior probabilities to target datasets in S3.

For each target horizon this script writes two sibling dataset sections:

1. Datasets that contain existing old price-HMM probabilities plus new
   ADA 12h/24h HMM probabilities.
2. Datasets that contain only the new ADA 12h/24h HMM probabilities.

All inputs and outputs are S3 objects.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

from build_price_feature_day import load_s3_parquet, make_s3_client
from prepare_ada_hmm_12h_24h_dataset import HMM_FEATURE_COLUMNS


S3_BUCKET = "binance-data-downloader"
DEFAULT_MODEL_PREFIX = "features/hmm_models/ada_hmm_12h_24h"
DEFAULT_HMM_DATASET_KEY = "features/hmm_dataset/ada_hmm_12h_24h/interval=1m/data.parquet"
DEFAULT_TARGET_PREFIXES = ("dataset_target_10", "dataset_target_20", "dataset_target_30")
DEFAULT_SPLIT_FILES = ("dataset.parquet", "train.parquet", "test.parquet")
DEFAULT_OLD_HMM_SECTION = "with_price_hmm_n4"
DEFAULT_OLD_AND_NEW_SECTION = "with_price_hmm_n4_and_ada_hmm_12h_24h"
DEFAULT_NEW_ONLY_SECTION = "with_ada_hmm_12h_24h"
DEFAULT_NEW_COLUMN_PREFIX = "ada_hmm_12h_24h"


def _list_s3_keys(s3: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        keys.extend(item["Key"] for item in page.get("Contents", []))
    return keys


def _latest_run_id(s3: Any, bucket: str, model_prefix: str) -> str:
    keys = _list_s3_keys(s3, bucket, f"{model_prefix.strip('/')}/runs")
    run_ids = sorted(
        {
            key.split("/run_id=", 1)[1].split("/", 1)[0]
            for key in keys
            if "/run_id=" in key
        }
    )
    if not run_ids:
        raise FileNotFoundError(
            f"No model runs found under s3://{bucket}/{model_prefix}/runs/"
        )
    return run_ids[-1]


def _s3_exists(s3: Any, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def _read_s3_bytes(s3: Any, bucket: str, key: str) -> bytes:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def _write_s3_bytes(
    s3: Any,
    bucket: str,
    key: str,
    payload: bytes,
    content_type: str,
    metadata: dict[str, str] | None = None,
) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType=content_type,
        Metadata=metadata or {},
    )


def _write_s3_parquet(
    s3: Any,
    bucket: str,
    key: str,
    frame: pd.DataFrame,
    metadata: dict[str, str] | None = None,
) -> int:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, engine="pyarrow", compression="zstd")
    payload = buffer.getvalue()
    _write_s3_bytes(
        s3,
        bucket,
        key,
        payload,
        "application/vnd.apache.parquet",
        metadata=metadata,
    )
    return len(payload)


def _load_s3_bundle(s3: Any, bucket: str, key: str) -> dict[str, Any]:
    bundle = joblib.load(io.BytesIO(_read_s3_bytes(s3, bucket, key)))
    required = {"model", "scaler", "features"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"HMM bundle is missing keys: {sorted(missing)}")
    return bundle


def _probability_columns(n_components: int, prefix: str) -> list[str]:
    return [f"{prefix}_prob_state_{state_id}" for state_id in range(n_components)]


def build_probability_frame(
    hmm_dataset: pd.DataFrame,
    bundle: dict[str, Any],
    column_prefix: str,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    model = bundle["model"]
    scaler = bundle["scaler"]
    features = list(bundle["features"])
    missing = [feature for feature in features if feature not in hmm_dataset.columns]
    if missing:
        raise ValueError(f"HMM probability source is missing features: {missing}")

    frame = hmm_dataset.loc[:, ["timestamp", *features]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("HMM probability source contains invalid timestamps")
    if frame["timestamp"].duplicated().any():
        raise ValueError("HMM probability source contains duplicate timestamps")

    feature_frame = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(feature_frame.to_numpy(dtype="float64", copy=False)).all(axis=1)
    valid_rows = int(finite_mask.sum())
    probability_columns = _probability_columns(model.n_components, column_prefix)

    probabilities = np.full((len(frame), model.n_components), np.nan, dtype="float32")
    if valid_rows:
        scaled = scaler.transform(feature_frame.loc[finite_mask].to_numpy(dtype="float64"))
        probabilities[finite_mask] = model.predict_proba(scaled).astype("float32", copy=False)

    result = frame.loc[:, ["timestamp"]].copy()
    for idx, column in enumerate(probability_columns):
        result[column] = probabilities[:, idx]
    result = result.sort_values("timestamp").reset_index(drop=True)

    probability_sums = probabilities[finite_mask].sum(axis=1) if valid_rows else np.array([])
    diagnostics: dict[str, int | float] = {
        "probability_source_rows": int(len(frame)),
        "valid_probability_rows": valid_rows,
        "invalid_probability_rows": int(len(frame) - valid_rows),
        "probability_columns": len(probability_columns),
    }
    if len(probability_sums):
        diagnostics["min_probability_sum"] = float(probability_sums.min())
        diagnostics["max_probability_sum"] = float(probability_sums.max())
    else:
        diagnostics["min_probability_sum"] = float("nan")
        diagnostics["max_probability_sum"] = float("nan")
    return result, diagnostics


def add_probabilities_to_target(
    target: pd.DataFrame,
    probabilities: pd.DataFrame,
    probability_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    if "timestamp" not in target.columns:
        raise ValueError("Target dataset is missing timestamp column")
    collisions = [column for column in probability_columns if column in target.columns]
    if collisions:
        raise ValueError(f"Output probability columns already exist: {collisions}")

    frame = target.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("Target dataset contains invalid timestamps")
    if frame["timestamp"].duplicated().any():
        raise ValueError("Target dataset contains duplicate timestamps")

    merged = frame.merge(
        probabilities.loc[:, ["timestamp", *probability_columns]],
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    complete_rows = int(merged.loc[:, probability_columns].notna().all(axis=1).sum())
    return merged, {
        "target_rows": int(len(frame)),
        "rows_with_new_hmm_probabilities": complete_rows,
        "rows_without_new_hmm_probabilities": int(len(frame) - complete_rows),
    }


def _output_key(target_prefix: str, section: str, split_file: str) -> str:
    return f"{target_prefix.strip('/')}/{section.strip('/')}/{split_file}"


def _input_key(target_prefix: str, section: str | None, split_file: str) -> str:
    if section:
        return f"{target_prefix.strip('/')}/{section.strip('/')}/{split_file}"
    return f"{target_prefix.strip('/')}/{split_file}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add new ADA HMM probabilities to target datasets in S3"
    )
    parser.add_argument("--bucket", default=os.getenv("YC_BUCKET", S3_BUCKET))
    parser.add_argument("--model-prefix", default=DEFAULT_MODEL_PREFIX)
    parser.add_argument("--run-id", help="Model run_id; defaults to latest under model prefix")
    parser.add_argument("--bundle-key", help="Explicit S3 key for model_bundle.joblib")
    parser.add_argument("--hmm-dataset-key", default=DEFAULT_HMM_DATASET_KEY)
    parser.add_argument("--target-prefixes", nargs="+", default=list(DEFAULT_TARGET_PREFIXES))
    parser.add_argument("--split-files", nargs="+", default=list(DEFAULT_SPLIT_FILES))
    parser.add_argument("--old-hmm-section", default=DEFAULT_OLD_HMM_SECTION)
    parser.add_argument("--old-and-new-section", default=DEFAULT_OLD_AND_NEW_SECTION)
    parser.add_argument("--new-only-section", default=DEFAULT_NEW_ONLY_SECTION)
    parser.add_argument("--new-column-prefix", default=DEFAULT_NEW_COLUMN_PREFIX)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    s3 = make_s3_client()
    run_id = args.run_id or (
        None if args.bundle_key else _latest_run_id(s3, args.bucket, args.model_prefix)
    )
    bundle_key = args.bundle_key or (
        f"{args.model_prefix.strip('/')}/runs/run_id={run_id}/model_bundle.joblib"
    )

    print(f"Loading new HMM bundle: s3://{args.bucket}/{bundle_key}", flush=True)
    bundle = _load_s3_bundle(s3, args.bucket, bundle_key)
    model = bundle["model"]
    probability_columns = _probability_columns(model.n_components, args.new_column_prefix)
    print(
        f"Loaded new HMM: n_components={model.n_components}, "
        f"features={list(bundle['features'])}",
        flush=True,
    )

    print(f"Loading new HMM probability source: s3://{args.bucket}/{args.hmm_dataset_key}")
    hmm_dataset = load_s3_parquet(s3, args.bucket, args.hmm_dataset_key)
    if hmm_dataset is None:
        raise FileNotFoundError(f"s3://{args.bucket}/{args.hmm_dataset_key}")
    probabilities, probability_diagnostics = build_probability_frame(
        hmm_dataset,
        bundle,
        args.new_column_prefix,
    )
    del hmm_dataset
    print(
        "Prepared probability frame: "
        f"rows={len(probabilities):,}, columns={probability_columns}, "
        f"diagnostics={probability_diagnostics}",
        flush=True,
    )

    manifest_base: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": args.bucket,
        "model_prefix": args.model_prefix,
        "run_id": run_id,
        "bundle_key": bundle_key,
        "hmm_dataset_key": args.hmm_dataset_key,
        "new_hmm_features": list(bundle["features"]),
        "new_probability_columns": probability_columns,
        "probability_diagnostics": probability_diagnostics,
        "outputs": [],
    }

    output_specs = (
        {
            "name": "old_and_new",
            "input_section": args.old_hmm_section,
            "output_section": args.old_and_new_section,
        },
        {
            "name": "new_only",
            "input_section": None,
            "output_section": args.new_only_section,
        },
    )

    for target_prefix in args.target_prefixes:
        for spec in output_specs:
            for split_file in args.split_files:
                input_key = _input_key(target_prefix, spec["input_section"], split_file)
                output_key = _output_key(target_prefix, spec["output_section"], split_file)
                if not args.overwrite and _s3_exists(s3, args.bucket, output_key):
                    raise FileExistsError(
                        f"Output already exists: s3://{args.bucket}/{output_key}. "
                        "Use --overwrite to replace it."
                    )

                print(f"Loading s3://{args.bucket}/{input_key}", flush=True)
                target = load_s3_parquet(s3, args.bucket, input_key)
                if target is None:
                    raise FileNotFoundError(f"s3://{args.bucket}/{input_key}")
                augmented, merge_diagnostics = add_probabilities_to_target(
                    target,
                    probabilities,
                    probability_columns,
                )
                del target

                output_record = {
                    "dataset_kind": spec["name"],
                    "target_prefix": target_prefix,
                    "split_file": split_file,
                    "input_key": input_key,
                    "output_key": output_key,
                    "input_columns": int(augmented.shape[1] - len(probability_columns)),
                    "output_columns": int(augmented.shape[1]),
                    **merge_diagnostics,
                }
                if args.dry_run:
                    print(f"DRY RUN {output_record}", flush=True)
                else:
                    size = _write_s3_parquet(
                        s3,
                        args.bucket,
                        output_key,
                        augmented,
                        metadata={
                            "source-key": input_key,
                            "new-hmm-bundle-key": bundle_key,
                            "new-hmm-column-prefix": args.new_column_prefix,
                            "dataset-kind": spec["name"],
                            "rows": str(len(augmented)),
                            "columns": str(augmented.shape[1]),
                        },
                    )
                    output_record["size_bytes"] = size
                    print(
                        f"Uploaded s3://{args.bucket}/{output_key} "
                        f"rows={len(augmented):,} columns={augmented.shape[1]} "
                        f"size_mib={size / 1024**2:.2f}",
                        flush=True,
                    )
                manifest_base["outputs"].append(output_record)
                del augmented

        if not args.dry_run:
            for spec in output_specs:
                output_section = spec["output_section"]
                output_prefix = f"{target_prefix.strip('/')}/{output_section.strip('/')}"
                manifest = {
                    **manifest_base,
                    "output_section": output_section,
                    "dataset_kind": spec["name"],
                    "outputs": [
                        row
                        for row in manifest_base["outputs"]
                        if row["output_key"].startswith(output_prefix + "/")
                    ],
                }
                manifest_key = f"{output_prefix}/ada_hmm_12h_24h_manifest.json"
                _write_s3_bytes(
                    s3,
                    args.bucket,
                    manifest_key,
                    json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
                    "application/json",
                    metadata={
                        "dataset-kind": spec["name"],
                        "new-hmm-column-prefix": args.new_column_prefix,
                    },
                )
                print(f"Uploaded manifest s3://{args.bucket}/{manifest_key}", flush=True)

    print("Completed target dataset HMM probability augmentation", flush=True)


if __name__ == "__main__":
    main()
