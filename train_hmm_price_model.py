"""Train and save one production GaussianHMM price model.

The default configuration matches the selected refined-grid price model:
full covariance, 4 regimes, random_state=15.  The saved bundle contains the
fitted HMM, fitted StandardScaler, feature list, and metadata needed for later
state inference.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for thread_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_env_var, "1")

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from build_price_feature_day import make_s3_client
from train_hmm_grid_search import INPUT_KEY, MODEL_GROUPS, S3_BUCKET
from train_hmm_refined_full_grid import _duration_stats, _state_duration_distribution


DEFAULT_OUTPUT_PREFIX = "features/hmm_models/price"
DEFAULT_LOCAL_OUTPUT_DIR = Path("analysis/hmm_grid_search/outputs/price_hmm_n4")


def _read_s3_parquet(s3: Any, bucket: str, key: str) -> pd.DataFrame:
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


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
) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, engine="pyarrow", compression="zstd")
    _write_s3_bytes(
        s3,
        bucket,
        key,
        buffer.getvalue(),
        "application/vnd.apache.parquet",
        metadata=metadata,
    )


def _model_parameter_count(n_components: int, n_features: int) -> int:
    start_probability = n_components - 1
    transition_matrix = n_components * (n_components - 1)
    means = n_components * n_features
    covariances = n_components * n_features * (n_features + 1) // 2
    return start_probability + transition_matrix + means + covariances


def _prepare_price_matrix(dataset: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, int]:
    features = MODEL_GROUPS["price"]
    missing = [feature for feature in features if feature not in dataset.columns]
    if missing:
        raise ValueError(f"Training dataset is missing price features: {missing}")

    frame = dataset.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if frame.empty:
        raise ValueError("Price feature frame has no complete finite rows")

    values = frame.to_numpy(dtype="float64", copy=False)
    scaler = StandardScaler()
    x = scaler.fit_transform(values).astype("float64", copy=False)
    rows_before_dropna = len(dataset)
    del frame, values
    gc.collect()
    return x, scaler, rows_before_dropna


def _build_result(
    model: GaussianHMM,
    x: np.ndarray,
    args: argparse.Namespace,
    rows_before_dropna: int,
    train_seconds: float,
) -> dict[str, Any]:
    rows, n_features = x.shape
    log_likelihood = float(model.score(x))
    states = model.predict(x)
    occupancy = np.bincount(states, minlength=args.n_components) / len(states)
    durations = _state_duration_distribution(states, args.n_components)
    duration_stats = _duration_stats(durations)
    parameter_count = _model_parameter_count(args.n_components, n_features)

    return {
        "model_group": "price",
        "features": json.dumps(list(MODEL_GROUPS["price"]), ensure_ascii=False),
        "n_components": args.n_components,
        "covariance_type": args.covariance_type,
        "random_state": args.random_state,
        "algorithm": "viterbi",
        "n_iter_requested": args.n_iter,
        "tol": args.tol,
        "n_rows_before_dropna": rows_before_dropna,
        "n_rows": rows,
        "n_features": n_features,
        "train_seconds": train_seconds,
        "n_iter": int(model.monitor_.iter),
        "converged": bool(model.monitor_.converged),
        "log_likelihood": log_likelihood,
        "aic": float(2 * parameter_count - 2 * log_likelihood),
        "bic": float(parameter_count * math.log(rows) - 2 * log_likelihood),
        "state_occupancy": json.dumps(occupancy.tolist()),
        "state_duration_distribution": json.dumps(durations),
        "state_duration_stats": json.dumps(duration_stats),
        "state_mean_duration": json.dumps(
            [state_stats["mean"] for state_stats in duration_stats]
        ),
        "transition_matrix": json.dumps(model.transmat_.tolist()),
        "state_means": json.dumps(model.means_.tolist()),
        "state_covariances": json.dumps(model.covars_.tolist()),
        "success": True,
        "error": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save one price GaussianHMM")
    parser.add_argument("--bucket", default=os.getenv("YC_BUCKET", S3_BUCKET))
    parser.add_argument("--input-key", default=INPUT_KEY)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("price_full_n4_%Y%m%d_%H%M%S"),
        help="Stable id for model artifact paths",
    )
    parser.add_argument("--local-output-dir", type=Path, default=DEFAULT_LOCAL_OUTPUT_DIR)
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--covariance-type", default="full", choices=("full", "diag"))
    parser.add_argument("--random-state", type=int, default=15)
    parser.add_argument("--n-iter", type=int, default=400)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--skip-s3-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    s3 = make_s3_client()

    print(f"Loading train dataset: s3://{args.bucket}/{args.input_key}", flush=True)
    dataset = _read_s3_parquet(s3, args.bucket, args.input_key)
    print(f"Loaded rows={len(dataset):,} columns={len(dataset.columns):,}", flush=True)

    x, scaler, rows_before_dropna = _prepare_price_matrix(dataset)
    del dataset
    gc.collect()
    print(
        f"Prepared price matrix: rows={x.shape[0]:,} features={x.shape[1]}",
        flush=True,
    )

    model = GaussianHMM(
        n_components=args.n_components,
        covariance_type=args.covariance_type,
        algorithm="viterbi",
        n_iter=args.n_iter,
        tol=args.tol,
        random_state=args.random_state,
    )
    started = time.perf_counter()
    model.fit(x)
    train_seconds = time.perf_counter() - started
    result = _build_result(model, x, args, rows_before_dropna, train_seconds)

    metadata = {
        "model_group": "price",
        "n_components": args.n_components,
        "covariance_type": args.covariance_type,
        "random_state": args.random_state,
        "features": list(MODEL_GROUPS["price"]),
        "input_key": args.input_key,
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    bundle = {
        "model": model,
        "scaler": scaler,
        "features": list(MODEL_GROUPS["price"]),
        "metadata": metadata,
    }

    local_run_dir = args.local_output_dir / f"run_id={args.run_id}"
    local_run_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = local_run_dir / "model_bundle.joblib"
    metadata_path = local_run_dir / "metadata.json"
    result_path = local_run_dir / "result.parquet"

    s3_run_prefix = f"{args.output_prefix.strip('/')}/runs/run_id={args.run_id}"
    result.update(
        {
            "local_bundle_path": str(bundle_path),
            "local_metadata_path": str(metadata_path),
            "local_result_path": str(result_path),
            "s3_bundle_key": f"{s3_run_prefix}/model_bundle.joblib",
            "s3_metadata_key": f"{s3_run_prefix}/metadata.json",
            "s3_result_key": f"{s3_run_prefix}/result.parquet",
        }
    )
    metadata["result"] = result

    joblib.dump(bundle, bundle_path, compress=3)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([result]).to_parquet(result_path, index=False, engine="pyarrow", compression="zstd")

    print(f"Saved local bundle: {bundle_path}", flush=True)
    print(f"Saved local metadata: {metadata_path}", flush=True)
    print(f"Saved local result: {result_path}", flush=True)

    if not args.skip_s3_upload:
        common_metadata = {
            "model-group": "price",
            "n-components": str(args.n_components),
            "covariance-type": args.covariance_type,
            "random-state": str(args.random_state),
            "run-id": args.run_id,
        }
        _write_s3_bytes(
            s3,
            args.bucket,
            result["s3_bundle_key"],
            bundle_path.read_bytes(),
            "application/octet-stream",
            metadata=common_metadata,
        )
        _write_s3_bytes(
            s3,
            args.bucket,
            result["s3_metadata_key"],
            metadata_path.read_bytes(),
            "application/json",
            metadata=common_metadata,
        )
        _write_s3_parquet(
            s3,
            args.bucket,
            result["s3_result_key"],
            pd.DataFrame([result]),
            metadata=common_metadata,
        )
        print(f"Uploaded bundle: s3://{args.bucket}/{result['s3_bundle_key']}", flush=True)
        print(f"Uploaded metadata: s3://{args.bucket}/{result['s3_metadata_key']}", flush=True)
        print(f"Uploaded result: s3://{args.bucket}/{result['s3_result_key']}", flush=True)

    print(
        "Done: "
        f"BIC={result['bic']:.2f}, AIC={result['aic']:.2f}, "
        f"LL={result['log_likelihood']:.2f}, converged={result['converged']}, "
        f"n_iter={result['n_iter']}, train_seconds={result['train_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
