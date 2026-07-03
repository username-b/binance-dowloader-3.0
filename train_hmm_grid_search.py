"""Train Gaussian HMM grid-search models on selected feature groups.

The script loads the training dataset once from S3, standardizes each feature
group, trains every HMM configuration independently in a process pool, stores
each result immediately, and writes one combined parquet table at the end.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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

import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from build_price_feature_day import make_s3_client


S3_BUCKET = "binance-data-downloader"
INPUT_KEY = "features/unified_dataset/splits/unified_dataset_train.parquet"
OUTPUT_PREFIX = "features/hmm_grid_search/results"
DEFAULT_WORKER_RESERVE = 4
DEFAULT_WORKER_CAP = 28
N_COMPONENTS = (2, 3, 4, 5, 6)
COVARIANCE_TYPES = ("diag", "full")
RANDOM_STATES = tuple(range(20))

MODEL_GROUPS: dict[str, tuple[str, ...]] = {
    "price": (
        "ada_log_return_lag_1",
        "ada_cum_return_30m",
        "ada_return_std_20m",
        "ada_kaufman_efficiency_90m",
        "ada_close_position_30m",
    ),
    "flow": (
        "ada_volume_zscore_60m",
        "ada_trades_per_minute_30m",
        "ada_aggression_delta_norm_20m",
        "ada_large_trade_aggression_delta_norm_20m",
        "ada_pressure_concentration_30m",
    ),
    "market": (
        "btc_log_return_30m",
        "btc_return_std_30m",
        "btc_kaufman_efficiency_60m",
        "time_of_day_sin",
        "time_of_day_cos",
    ),
}

_WORKER_ARRAY_PATHS: dict[str, str] = {}


@dataclass(frozen=True)
class HMMTask:
    model_group: str
    n_components: int
    covariance_type: str
    random_state: int
    result_key: str


def _read_s3_parquet(s3: Any, bucket: str, key: str) -> pd.DataFrame:
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


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
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="application/vnd.apache.parquet",
        Metadata=metadata or {},
    )
    return len(payload)


def _s3_exists(s3: Any, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def _list_result_keys(s3: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/result.parquet"):
                keys.append(key)
    return keys


def _result_key(output_prefix: str, run_id: str, task: HMMTask) -> str:
    return (
        f"{output_prefix.strip('/')}/runs/run_id={run_id}/models/"
        f"model_group={task.model_group}/"
        f"n_components={task.n_components}/"
        f"covariance_type={task.covariance_type}/"
        f"random_state={task.random_state}/result.parquet"
    )


def _final_key(output_prefix: str, run_id: str) -> str:
    return f"{output_prefix.strip('/')}/runs/run_id={run_id}/all_results.parquet"


def _validate_columns(dataset: pd.DataFrame) -> None:
    required = sorted({column for columns in MODEL_GROUPS.values() for column in columns})
    missing = [column for column in required if column not in dataset.columns]
    if missing:
        raise ValueError(f"Training dataset is missing required columns: {missing}")


def _prepare_group_arrays(dataset: pd.DataFrame, directory: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for group, features in MODEL_GROUPS.items():
        frame = dataset.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if frame.empty:
            raise ValueError(f"Feature group {group!r} has no complete finite rows")

        values = frame.to_numpy(dtype="float64", copy=False)
        scaled = StandardScaler().fit_transform(values).astype("float32", copy=False)
        path = directory / f"{group}.npy"
        np.save(path, scaled, allow_pickle=False)
        paths[group] = str(path)
        print(
            f"Prepared {group}: rows={scaled.shape[0]:,} features={scaled.shape[1]}",
            flush=True,
        )
        del frame, values, scaled
        gc.collect()
    return paths


def _build_tasks(output_prefix: str, run_id: str) -> list[HMMTask]:
    tasks: list[HMMTask] = []
    for model_group in MODEL_GROUPS:
        for n_components in N_COMPONENTS:
            for covariance_type in COVARIANCE_TYPES:
                for random_state in RANDOM_STATES:
                    partial_task = HMMTask(
                        model_group,
                        n_components,
                        covariance_type,
                        random_state,
                        "",
                    )
                    tasks.append(
                        HMMTask(
                            model_group,
                            n_components,
                            covariance_type,
                            random_state,
                            _result_key(output_prefix, run_id, partial_task),
                        )
                    )
    return tasks


def _init_worker(array_paths: dict[str, str]) -> None:
    global _WORKER_ARRAY_PATHS
    _WORKER_ARRAY_PATHS = array_paths


def _resolve_max_workers(
    requested_workers: int | None,
    worker_reserve: int,
    worker_cap: int | None,
) -> int:
    if worker_reserve < 0:
        raise ValueError("--worker-reserve must not be negative")
    if worker_cap is not None and worker_cap < 1:
        raise ValueError("--worker-cap must be positive when provided")

    detected_cpu = os.cpu_count() or 1
    automatic_workers = max(1, detected_cpu - worker_reserve)
    if worker_cap is not None:
        automatic_workers = min(automatic_workers, worker_cap)

    if requested_workers is None:
        return automatic_workers
    if requested_workers < 1:
        raise ValueError("--max-workers must be positive")
    if requested_workers > automatic_workers:
        print(
            f"Requested workers={requested_workers} exceeds safe limit "
            f"{automatic_workers} for cpu={detected_cpu}, reserve={worker_reserve}, "
            f"cap={worker_cap}; using {automatic_workers}",
            flush=True,
        )
    return min(requested_workers, automatic_workers)


def _model_parameter_count(n_components: int, n_features: int, covariance_type: str) -> int:
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


def _state_mean_durations(states: np.ndarray, n_components: int) -> list[float | None]:
    if len(states) == 0:
        return [None] * n_components

    durations: list[list[int]] = [[] for _ in range(n_components)]
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

    return [
        float(np.mean(state_durations)) if state_durations else None
        for state_durations in durations
    ]


def _base_result(task: HMMTask, rows: int | None, error: str | None = None) -> dict[str, Any]:
    return {
        "model_group": task.model_group,
        "features": json.dumps(list(MODEL_GROUPS[task.model_group]), ensure_ascii=False),
        "n_components": task.n_components,
        "covariance_type": task.covariance_type,
        "random_state": task.random_state,
        "n_rows": rows,
        "train_seconds": None,
        "n_iter": None,
        "converged": None,
        "success": error is None,
        "error": error,
        "log_likelihood": None,
        "aic": None,
        "bic": None,
        "state_occupancy": None,
        "state_mean_duration": None,
        "transition_matrix": None,
        "result_key": task.result_key,
    }


def _train_one(task: HMMTask) -> dict[str, Any]:
    started = time.perf_counter()
    x: np.ndarray | None = None
    try:
        x = np.load(_WORKER_ARRAY_PATHS[task.model_group], mmap_mode="r")
        rows, n_features = x.shape
        model = GaussianHMM(
            n_components=task.n_components,
            covariance_type=task.covariance_type,
            algorithm="viterbi",
            n_iter=400,
            tol=1e-6,
            random_state=task.random_state,
        )
        model.fit(x)
        log_likelihood = float(model.score(x))
        states = model.predict(x)
        occupancy = np.bincount(states, minlength=task.n_components) / len(states)
        parameter_count = _model_parameter_count(
            task.n_components,
            n_features,
            task.covariance_type,
        )
        result = _base_result(task, rows)
        result.update(
            {
                "train_seconds": time.perf_counter() - started,
                "n_iter": int(model.monitor_.iter),
                "converged": bool(model.monitor_.converged),
                "log_likelihood": log_likelihood,
                "aic": float(2 * parameter_count - 2 * log_likelihood),
                "bic": float(parameter_count * math.log(rows) - 2 * log_likelihood),
                "state_occupancy": json.dumps(occupancy.tolist()),
                "state_mean_duration": json.dumps(
                    _state_mean_durations(states, task.n_components)
                ),
                "transition_matrix": json.dumps(model.transmat_.tolist()),
            }
        )
        return result
    except Exception as exc:
        rows = int(x.shape[0]) if x is not None else None
        result = _base_result(task, rows, error=repr(exc))
        result["train_seconds"] = time.perf_counter() - started
        return result
    finally:
        mmap = getattr(x, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _write_single_result(s3: Any, bucket: str, result: dict[str, Any]) -> int:
    frame = pd.DataFrame([result])
    return _write_s3_parquet(
        s3,
        bucket,
        result["result_key"],
        frame,
        metadata={
            "model-group": str(result["model_group"]),
            "success": str(bool(result["success"])).lower(),
        },
    )


def _combine_results(s3: Any, bucket: str, output_prefix: str, run_id: str) -> pd.DataFrame:
    model_prefix = f"{output_prefix.strip('/')}/runs/run_id={run_id}/models"
    keys = _list_result_keys(s3, bucket, model_prefix)
    if not keys:
        raise FileNotFoundError(f"No result parquet files found under s3://{bucket}/{model_prefix}")

    frames = [_read_s3_parquet(s3, bucket, key) for key in sorted(keys)]
    combined = pd.concat(frames, ignore_index=True)
    sort_columns = ["model_group", "n_components", "covariance_type", "random_state"]
    combined = combined.sort_values(sort_columns).reset_index(drop=True)
    _write_s3_parquet(
        s3,
        bucket,
        _final_key(output_prefix, run_id),
        combined,
        metadata={
            "run-id": run_id,
            "rows": str(len(combined)),
            "successes": str(int(combined["success"].sum())),
            "failures": str(int((~combined["success"]).sum())),
        },
    )
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a full GaussianHMM hyperparameter search for selected groups"
    )
    parser.add_argument("--bucket", default=os.getenv("YC_BUCKET", S3_BUCKET))
    parser.add_argument("--input-key", default=INPUT_KEY)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        help="Stable id for resume; use the same value after a crash",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Explicit worker count. By default it is computed as "
            "min(cpu_count - --worker-reserve, --worker-cap)."
        ),
    )
    parser.add_argument(
        "--worker-reserve",
        type=int,
        default=DEFAULT_WORKER_RESERVE,
        help="vCPU count to leave for the OS and S3/parquet overhead",
    )
    parser.add_argument(
        "--worker-cap",
        type=int,
        default=DEFAULT_WORKER_CAP,
        help="Upper bound for automatically selected workers; use 0 to disable",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain configurations even if their result parquet already exists",
    )
    parser.add_argument(
        "--skip-final-combine",
        action="store_true",
        help="Do not write the final all_results.parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker_cap = None if args.worker_cap == 0 else args.worker_cap
    max_workers = _resolve_max_workers(
        args.max_workers,
        args.worker_reserve,
        worker_cap,
    )

    s3 = make_s3_client()
    print(f"Loading train dataset: s3://{args.bucket}/{args.input_key}", flush=True)
    dataset = _read_s3_parquet(s3, args.bucket, args.input_key)
    _validate_columns(dataset)
    print(f"Loaded rows={len(dataset):,} columns={len(dataset.columns):,}", flush=True)

    run_root = f"{args.output_prefix.strip('/')}/runs/run_id={args.run_id}"
    print(f"Run output: s3://{args.bucket}/{run_root}/", flush=True)

    with tempfile.TemporaryDirectory(prefix="hmm_grid_") as temp_dir:
        array_paths = _prepare_group_arrays(dataset, Path(temp_dir))
        del dataset
        gc.collect()

        all_tasks = _build_tasks(args.output_prefix, args.run_id)
        if args.overwrite:
            tasks = all_tasks
        else:
            tasks = [
                task
                for task in all_tasks
                if not _s3_exists(s3, args.bucket, task.result_key)
            ]
        skipped = len(all_tasks) - len(tasks)
        print(
            f"Scheduled {len(tasks)} configs, skipped existing {skipped}, "
            f"workers={max_workers}, cpu={os.cpu_count() or 1}, "
            f"reserve={args.worker_reserve}, cap={worker_cap}",
            flush=True,
        )

        completed = succeeded = failed = 0
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(array_paths,),
        ) as executor:
            future_to_task = {executor.submit(_train_one, task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = _base_result(task, None, error=repr(exc))
                size = _write_single_result(s3, args.bucket, result)
                completed += 1
                succeeded += int(bool(result["success"]))
                failed += int(not bool(result["success"]))
                status = "success" if result["success"] else "failed"
                print(
                    f"[{completed}/{len(tasks)}] {status} "
                    f"{task.model_group} n={task.n_components} "
                    f"cov={task.covariance_type} seed={task.random_state} "
                    f"seconds={result['train_seconds']} size_kib={size / 1024:.1f}",
                    flush=True,
                )

    if not args.skip_final_combine:
        combined = _combine_results(s3, args.bucket, args.output_prefix, args.run_id)
        print(
            f"Final table: s3://{args.bucket}/{_final_key(args.output_prefix, args.run_id)} "
            f"rows={len(combined)} successes={int(combined['success'].sum())} "
            f"failures={int((~combined['success']).sum())}",
            flush=True,
        )
    print(f"Completed run_id={args.run_id}")


if __name__ == "__main__":
    main()
