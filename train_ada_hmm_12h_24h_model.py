"""Train and save the selected ADA 12h/24h GaussianHMM model to S3."""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import time
from datetime import datetime, timezone
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
from sklearn.preprocessing import StandardScaler

from build_price_feature_day import make_s3_client
from prepare_ada_hmm_12h_24h_dataset import HMM_FEATURE_COLUMNS


S3_BUCKET = "binance-data-downloader"
DEFAULT_INPUT_KEY = "features/hmm_dataset/ada_hmm_12h_24h/interval=1m/data.parquet"
DEFAULT_OUTPUT_PREFIX = "features/hmm_models/ada_hmm_12h_24h"


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


def _model_parameter_count(
    n_components: int,
    n_features: int,
    covariance_type: str,
) -> int:
    start_probability = n_components - 1
    transition_matrix = n_components * (n_components - 1)
    means = n_components * n_features
    if covariance_type == "diag":
        covariances = n_components * n_features
    elif covariance_type == "full":
        covariances = n_components * n_features * (n_features + 1) // 2
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")
    return start_probability + transition_matrix + means + covariances


def _state_duration_distribution(
    states: np.ndarray,
    n_components: int,
) -> list[list[int]]:
    durations: list[list[int]] = [[] for _ in range(n_components)]
    if len(states) == 0:
        return durations

    current_state = int(states[0])
    current_length = 1
    for state in states[1:]:
        state = int(state)
        if state == current_state:
            current_length += 1
        else:
            durations[current_state].append(current_length)
            current_state = state
            current_length = 1
    durations[current_state].append(current_length)
    return durations


def _duration_stats(durations: list[list[int]]) -> list[dict[str, float | int | None]]:
    result: list[dict[str, float | int | None]] = []
    quantiles = (0.5, 0.75, 0.9, 0.95, 0.99)
    for state_durations in durations:
        if not state_durations:
            result.append(
                {
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "p50": None,
                    "p75": None,
                    "p90": None,
                    "p95": None,
                    "p99": None,
                    "max": None,
                }
            )
            continue
        values = np.asarray(state_durations, dtype="float64")
        stats = {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": int(values.min()),
            "max": int(values.max()),
        }
        stats.update(
            {
                f"p{int(quantile * 100)}": float(np.quantile(values, quantile))
                for quantile in quantiles
            }
        )
        result.append(stats)
    return result


def _prepare_training_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", *HMM_FEATURE_COLUMNS}
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"Training dataset is missing columns: {sorted(missing)}")

    frame = dataset.loc[:, ["timestamp", *HMM_FEATURE_COLUMNS]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("Training dataset contains invalid timestamps")
    if frame["timestamp"].duplicated().any():
        raise ValueError("Training dataset contains duplicate timestamps")

    for feature in HMM_FEATURE_COLUMNS:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if frame.empty:
        raise ValueError("Training feature frame has no complete finite rows")
    return frame.sort_values("timestamp").reset_index(drop=True)


def _prepare_matrix(training_frame: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    values = training_frame.loc[:, list(HMM_FEATURE_COLUMNS)].to_numpy(
        dtype="float64",
        copy=False,
    )
    scaler = StandardScaler()
    x = scaler.fit_transform(values).astype("float64", copy=False)
    return x, scaler


def _build_result(
    model: Any,
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
    parameter_count = _model_parameter_count(
        args.n_components,
        n_features,
        args.covariance_type,
    )

    return {
        "model_name": "ada_hmm_12h_24h",
        "features": json.dumps(list(HMM_FEATURE_COLUMNS), ensure_ascii=False),
        "n_components": args.n_components,
        "covariance_type": args.covariance_type,
        "algorithm": "viterbi",
        "random_state": args.random_state,
        "n_iter_requested": args.n_iter,
        "tol": args.tol,
        "min_covar": args.min_covar,
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
        "state_means_scaled": json.dumps(model.means_.tolist()),
        "state_covariances_scaled": json.dumps(model.covars_.tolist()),
        "success": True,
        "error": None,
    }


def _build_state_profiles(
    training_frame: pd.DataFrame,
    states: np.ndarray,
    n_components: int,
) -> pd.DataFrame:
    values = training_frame.loc[:, list(HMM_FEATURE_COLUMNS)]
    rows: list[dict[str, float | int | None]] = []
    for state in range(n_components):
        mask = states == state
        state_values = values.loc[mask]
        row: dict[str, float | int | None] = {
            "state": state,
            "rows": int(mask.sum()),
            "occupancy": float(mask.mean()),
        }
        for feature in HMM_FEATURE_COLUMNS:
            series = state_values[feature]
            row[f"{feature}_mean"] = float(series.mean()) if len(series) else None
            row[f"{feature}_std"] = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            row[f"{feature}_p05"] = float(series.quantile(0.05)) if len(series) else None
            row[f"{feature}_p50"] = float(series.quantile(0.50)) if len(series) else None
            row[f"{feature}_p95"] = float(series.quantile(0.95)) if len(series) else None
        rows.append(row)
    return pd.DataFrame(rows)


def _joblib_dumps(value: Any, compress: int = 3) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(value, buffer, compress=compress)
    return buffer.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ADA 12h/24h GaussianHMM and save artifacts to S3"
    )
    parser.add_argument("--bucket", default=os.getenv("YC_BUCKET", S3_BUCKET))
    parser.add_argument("--input-key", default=DEFAULT_INPUT_KEY)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("ada_hmm_12h_24h_%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--covariance-type", default="diag", choices=("diag", "full"))
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--min-covar", type=float, default=1e-2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-states",
        action="store_true",
        help="Do not save timestamp/state assignments",
    )
    return parser.parse_args()


def main() -> None:
    from hmmlearn.hmm import GaussianHMM

    args = parse_args()
    s3 = make_s3_client()

    print(f"Loading training dataset: s3://{args.bucket}/{args.input_key}", flush=True)
    dataset = _read_s3_parquet(s3, args.bucket, args.input_key)
    rows_before_dropna = len(dataset)
    training_frame = _prepare_training_frame(dataset)
    del dataset
    gc.collect()
    x, scaler = _prepare_matrix(training_frame)
    print(
        f"Prepared matrix: rows={x.shape[0]:,} features={x.shape[1]}",
        flush=True,
    )

    model = GaussianHMM(
        n_components=args.n_components,
        covariance_type=args.covariance_type,
        algorithm="viterbi",
        n_iter=args.n_iter,
        tol=args.tol,
        min_covar=args.min_covar,
        random_state=args.random_state,
    )
    started = time.perf_counter()
    model.fit(x)
    train_seconds = time.perf_counter() - started
    result = _build_result(model, x, args, rows_before_dropna, train_seconds)

    states = model.predict(x)
    profiles = _build_state_profiles(training_frame, states, args.n_components)
    states_frame = pd.DataFrame(
        {
            "timestamp": training_frame["timestamp"],
            "state": states.astype("int16"),
        }
    )

    run_prefix = f"{args.output_prefix.strip('/')}/runs/run_id={args.run_id}"
    result.update(
        {
            "s3_bundle_key": f"{run_prefix}/model_bundle.joblib",
            "s3_metadata_key": f"{run_prefix}/metadata.json",
            "s3_result_key": f"{run_prefix}/result.parquet",
            "s3_profiles_key": f"{run_prefix}/state_profiles.parquet",
            "s3_states_key": None if args.skip_states else f"{run_prefix}/train_states.parquet",
        }
    )
    metadata = {
        "model_name": "ada_hmm_12h_24h",
        "features": list(HMM_FEATURE_COLUMNS),
        "input_key": args.input_key,
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    bundle = {
        "model": model,
        "scaler": scaler,
        "features": list(HMM_FEATURE_COLUMNS),
        "metadata": metadata,
    }
    common_metadata = {
        "model-name": "ada_hmm_12h_24h",
        "run-id": args.run_id,
        "n-components": str(args.n_components),
        "covariance-type": args.covariance_type,
        "random-state": str(args.random_state),
    }

    _write_s3_bytes(
        s3,
        args.bucket,
        result["s3_bundle_key"],
        _joblib_dumps(bundle),
        "application/octet-stream",
        metadata=common_metadata,
    )
    _write_s3_bytes(
        s3,
        args.bucket,
        result["s3_metadata_key"],
        json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
        "application/json",
        metadata=common_metadata,
    )
    result_size = _write_s3_parquet(
        s3,
        args.bucket,
        result["s3_result_key"],
        pd.DataFrame([result]),
        metadata=common_metadata,
    )
    profiles_size = _write_s3_parquet(
        s3,
        args.bucket,
        result["s3_profiles_key"],
        profiles,
        metadata=common_metadata,
    )
    if not args.skip_states:
        states_size = _write_s3_parquet(
            s3,
            args.bucket,
            result["s3_states_key"],
            states_frame,
            metadata=common_metadata,
        )
    else:
        states_size = 0

    print(f"Uploaded bundle: s3://{args.bucket}/{result['s3_bundle_key']}", flush=True)
    print(f"Uploaded metadata: s3://{args.bucket}/{result['s3_metadata_key']}", flush=True)
    print(
        f"Uploaded result: s3://{args.bucket}/{result['s3_result_key']} "
        f"size_kib={result_size / 1024:.1f}",
        flush=True,
    )
    print(
        f"Uploaded profiles: s3://{args.bucket}/{result['s3_profiles_key']} "
        f"size_kib={profiles_size / 1024:.1f}",
        flush=True,
    )
    if not args.skip_states:
        print(
            f"Uploaded states: s3://{args.bucket}/{result['s3_states_key']} "
            f"size_mib={states_size / 1024**2:.2f}",
            flush=True,
        )
    print(
        "Done: "
        f"BIC={result['bic']:.2f}, AIC={result['aic']:.2f}, "
        f"LL={result['log_likelihood']:.2f}, converged={result['converged']}, "
        f"n_iter={result['n_iter']}, train_seconds={result['train_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
