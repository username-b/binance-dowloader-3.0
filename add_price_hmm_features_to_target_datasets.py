"""Add price-HMM posterior probabilities to target datasets in S3.

For every dataset_target_10/20/30 split file, this script loads the existing
features, applies the saved price GaussianHMM bundle, and writes a new sibling
section with four probability columns:

    price_hmm_n4_prob_state_0
    price_hmm_n4_prob_state_1
    price_hmm_n4_prob_state_2
    price_hmm_n4_prob_state_3

The original target datasets are left untouched.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

from build_price_feature_day import load_s3_parquet, make_s3_client
from train_hmm_grid_search import S3_BUCKET


DEFAULT_BUNDLE_PATH = Path(
    "analysis/hmm_grid_search/outputs/price_hmm_n4/"
    "run_id=price_full_n4_20260709_210000/model_bundle.joblib"
)
DEFAULT_TARGET_PREFIXES = ("dataset_target_10", "dataset_target_20", "dataset_target_30")
DEFAULT_SPLIT_FILES = ("dataset.parquet", "train.parquet", "test.parquet")
DEFAULT_OUTPUT_SECTION = "with_price_hmm_n4"


def _s3_exists(s3: Any, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


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


def _load_bundle(bundle_path: Path) -> dict[str, Any]:
    if not bundle_path.exists():
        raise FileNotFoundError(f"HMM bundle not found: {bundle_path}")
    bundle = joblib.load(bundle_path)
    required = {"model", "scaler", "features"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"HMM bundle is missing keys: {sorted(missing)}")
    return bundle


def _probability_columns(n_components: int, prefix: str) -> list[str]:
    return [f"{prefix}_prob_state_{state_id}" for state_id in range(n_components)]


def _add_hmm_probabilities(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    column_prefix: str,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    model = bundle["model"]
    scaler = bundle["scaler"]
    features = list(bundle["features"])
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError(f"Target dataset is missing HMM features: {missing}")

    probability_columns = _probability_columns(model.n_components, column_prefix)
    collisions = [column for column in probability_columns if column in frame.columns]
    if collisions:
        raise ValueError(f"Output columns already exist: {collisions}")

    feature_frame = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(feature_frame.to_numpy(dtype="float64", copy=False)).all(axis=1)

    probabilities = np.full(
        (len(frame), model.n_components),
        np.nan,
        dtype="float32",
    )
    valid_rows = int(finite_mask.sum())
    if valid_rows:
        scaled = scaler.transform(feature_frame.loc[finite_mask].to_numpy(dtype="float64"))
        probabilities[finite_mask] = model.predict_proba(scaled).astype("float32", copy=False)

    augmented = frame.copy()
    for idx, column in enumerate(probability_columns):
        augmented[column] = probabilities[:, idx]

    finite_probability_sums = probabilities[finite_mask].sum(axis=1) if valid_rows else np.array([])
    diagnostics: dict[str, int | float] = {
        "rows": int(len(frame)),
        "valid_hmm_rows": valid_rows,
        "invalid_hmm_rows": int(len(frame) - valid_rows),
        "probability_columns": len(probability_columns),
    }
    if len(finite_probability_sums):
        diagnostics["min_probability_sum"] = float(finite_probability_sums.min())
        diagnostics["max_probability_sum"] = float(finite_probability_sums.max())
    else:
        diagnostics["min_probability_sum"] = float("nan")
        diagnostics["max_probability_sum"] = float("nan")

    return augmented, diagnostics


def _target_output_prefix(input_prefix: str, output_section: str) -> str:
    return f"{input_prefix.strip('/')}/{output_section.strip('/')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add price HMM posterior probability features to target datasets"
    )
    parser.add_argument("--bucket", default=os.getenv("YC_BUCKET", S3_BUCKET))
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--target-prefixes", nargs="+", default=list(DEFAULT_TARGET_PREFIXES))
    parser.add_argument("--split-files", nargs="+", default=list(DEFAULT_SPLIT_FILES))
    parser.add_argument("--output-section", default=DEFAULT_OUTPUT_SECTION)
    parser.add_argument("--column-prefix", default="price_hmm_n4")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = _load_bundle(args.bundle_path)
    model = bundle["model"]
    features = list(bundle["features"])
    probability_columns = _probability_columns(model.n_components, args.column_prefix)

    print(
        "Loaded HMM bundle: "
        f"path={args.bundle_path}, n_components={model.n_components}, "
        f"features={features}",
        flush=True,
    )

    s3 = make_s3_client()
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": args.bucket,
        "bundle_path": str(args.bundle_path),
        "model_group": "price",
        "n_components": int(model.n_components),
        "covariance_type": str(model.covariance_type),
        "random_state": int(model.random_state) if model.random_state is not None else None,
        "hmm_features": features,
        "probability_columns": probability_columns,
        "output_section": args.output_section,
        "outputs": [],
    }

    for target_prefix in args.target_prefixes:
        output_prefix = _target_output_prefix(target_prefix, args.output_section)
        for split_file in args.split_files:
            input_key = f"{target_prefix.strip('/')}/{split_file}"
            output_key = f"{output_prefix}/{split_file}"

            if not args.overwrite and _s3_exists(s3, args.bucket, output_key):
                raise FileExistsError(
                    f"Output already exists: s3://{args.bucket}/{output_key}. "
                    "Use --overwrite to replace it."
                )

            print(f"Loading s3://{args.bucket}/{input_key}", flush=True)
            frame = load_s3_parquet(s3, args.bucket, input_key)
            if frame is None:
                raise FileNotFoundError(f"s3://{args.bucket}/{input_key}")

            augmented, diagnostics = _add_hmm_probabilities(
                frame,
                bundle,
                args.column_prefix,
            )
            del frame

            output_record = {
                "input_key": input_key,
                "output_key": output_key,
                "input_columns": int(augmented.shape[1] - len(probability_columns)),
                "output_columns": int(augmented.shape[1]),
                **diagnostics,
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
                        "hmm-model-group": "price",
                        "hmm-n-components": str(model.n_components),
                        "hmm-column-prefix": args.column_prefix,
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
            manifest["outputs"].append(output_record)
            del augmented

        if not args.dry_run:
            manifest_key = f"{output_prefix}/price_hmm_n4_manifest.json"
            target_manifest = {
                **manifest,
                "outputs": [
                    row
                    for row in manifest["outputs"]
                    if row["output_key"].startswith(output_prefix + "/")
                ],
            }
            _write_s3_bytes(
                s3,
                args.bucket,
                manifest_key,
                json.dumps(target_manifest, ensure_ascii=False, indent=2).encode("utf-8"),
                "application/json",
                metadata={
                    "hmm-model-group": "price",
                    "hmm-n-components": str(model.n_components),
                },
            )
            print(f"Uploaded manifest s3://{args.bucket}/{manifest_key}", flush=True)

    print("Completed HMM feature augmentation", flush=True)


if __name__ == "__main__":
    main()
