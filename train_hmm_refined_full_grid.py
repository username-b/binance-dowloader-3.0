"""Run a refined full-covariance GaussianHMM grid.

The refined grid uses the best random_state values from a completed broad run:

* select top N unique random_state values per model group among full-covariance
  models ranked by BIC;
* train full-covariance HMMs with n_components from 2 to 10;
* store richer regime diagnostics, including duration distributions, state
  means, state covariance matrices, and transition matrices.
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
from train_hmm_grid_search import (
    DEFAULT_WORKER_CAP,
    DEFAULT_WORKER_RESERVE,
    INPUT_KEY,
    MODEL_GROUPS,
    S3_BUCKET,
    _resolve_max_workers,
)


BROAD_OUTPUT_PREFIX = "features/hmm_grid_search/results"
REFINED_OUTPUT_PREFIX = "features/hmm_grid_search/refined_full_results"
N_COMPONENTS = tuple(range(2, 11))
TOP_RANDOM_STATES = 5

_WORKER_ARRAY_PATHS: dict[str, str] = {}


@dataclass(frozen=True)
class HMMTask:
    model_group: str
    n_components: int
    random_state: int
    selected_from_broad_bic: float | None
    selected_from_broad_n_components: int | None
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


def _list_s3_keys(s3: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        keys.extend(item["Key"] for item in page.get("Contents", []))
    return keys


def _latest_run_id(s3: Any, bucket: str, output_prefix: str) -> str:
    keys = _list_s3_keys(s3, bucket, f"{output_prefix.strip('/')}/runs")
    run_ids = sorted(
        {
            key.split("/run_id=", 1)[1].split("/", 1)[0]
            for key in keys
            if "/run_id=" in key
        }
    )
    if not run_ids:
        raise FileNotFoundError(f"No broad run ids found under s3://{bucket}/{output_prefix}/runs/")
    return run_ids[-1]


def _broad_all_results_key(output_prefix: str, run_id: str) -> str:
    return f"{output_prefix.strip('/')}/runs/run_id={run_id}/all_results.parquet"


def _refined_result_key(output_prefix: str, run_id: str, task: HMMTask) -> str:
    return (
        f"{output_prefix.strip('/')}/runs/run_id={run_id}/models/"
        f"model_group={task.model_group}/"
        f"n_components={task.n_components}/"
        "covariance_type=full/"
        f"random_state={task.random_state}/result.parquet"
    )


def _refined_final_key(output_prefix: str, run_id: str) -> str:
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


def _select_top_random_states(
    broad_results: pd.DataFrame,
    top_n: int,
) -> dict[str, list[dict[str, int | float]]]:
    required = {
        "model_group",
        "covariance_type",
        "random_state",
        "n_components",
        "bic",
        "success",
    }
    missing = required.difference(broad_results.columns)
    if missing:
        raise ValueError(f"Broad results are missing columns: {sorted(missing)}")

    frame = broad_results.copy()
    frame["success"] = frame["success"].astype(bool)
    frame["bic"] = pd.to_numeric(frame["bic"], errors="coerce")
    frame["random_state"] = pd.to_numeric(frame["random_state"], errors="coerce")
    frame["n_components"] = pd.to_numeric(frame["n_components"], errors="coerce")
    frame = frame[
        frame["success"]
        & frame["covariance_type"].eq("full")
        & frame["bic"].notna()
        & frame["random_state"].notna()
    ].sort_values("bic", ascending=True)

    selected: dict[str, list[dict[str, int | float]]] = {}
    for group in MODEL_GROUPS:
        group_frame = frame[frame["model_group"].eq(group)]
        states: list[dict[str, int | float]] = []
        seen: set[int] = set()
        for row in group_frame.itertuples(index=False):
            random_state = int(row.random_state)
            if random_state in seen:
                continue
            states.append(
                {
                    "random_state": random_state,
                    "broad_bic": float(row.bic),
                    "broad_n_components": int(row.n_components),
                }
            )
            seen.add(random_state)
            if len(states) == top_n:
                break
        if len(states) != top_n:
            raise ValueError(
                f"Only found {len(states)} unique full-covariance random_state values "
                f"for {group}, expected {top_n}"
            )
        selected[group] = states
    return selected


def _build_tasks(
    output_prefix: str,
    run_id: str,
    selected_states: dict[str, list[dict[str, int | float]]],
) -> list[HMMTask]:
    tasks: list[HMMTask] = []
    for model_group, states in selected_states.items():
        for n_components in N_COMPONENTS:
            for state in states:
                partial_task = HMMTask(
                    model_group=model_group,
                    n_components=n_components,
                    random_state=int(state["random_state"]),
                    selected_from_broad_bic=float(state["broad_bic"]),
                    selected_from_broad_n_components=int(state["broad_n_components"]),
                    result_key="",
                )
                tasks.append(
                    HMMTask(
                        model_group=partial_task.model_group,
                        n_components=partial_task.n_components,
                        random_state=partial_task.random_state,
                        selected_from_broad_bic=partial_task.selected_from_broad_bic,
                        selected_from_broad_n_components=partial_task.selected_from_broad_n_components,
                        result_key=_refined_result_key(output_prefix, run_id, partial_task),
                    )
                )
    return tasks


def _init_worker(array_paths: dict[str, str]) -> None:
    global _WORKER_ARRAY_PATHS
    _WORKER_ARRAY_PATHS = array_paths


def _model_parameter_count(n_components: int, n_features: int) -> int:
    start_probability = n_components - 1
    transition_matrix = n_components * (n_components - 1)
    means = n_components * n_features
    covariances = n_components * n_features * (n_features + 1) // 2
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


def _base_result(task: HMMTask, rows: int | None, error: str | None = None) -> dict[str, Any]:
    return {
        "model_group": task.model_group,
        "features": json.dumps(list(MODEL_GROUPS[task.model_group]), ensure_ascii=False),
        "n_components": task.n_components,
        "covariance_type": "full",
        "random_state": task.random_state,
        "selected_from_broad_bic": task.selected_from_broad_bic,
        "selected_from_broad_n_components": task.selected_from_broad_n_components,
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
        "state_duration_distribution": None,
        "state_duration_stats": None,
        "state_mean_duration": None,
        "transition_matrix": None,
        "state_means": None,
        "state_covariances": None,
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
            covariance_type="full",
            algorithm="viterbi",
            n_iter=400,
            tol=1e-6,
            random_state=task.random_state,
        )
        model.fit(x)
        log_likelihood = float(model.score(x))
        states = model.predict(x)
        occupancy = np.bincount(states, minlength=task.n_components) / len(states)
        durations = _state_duration_distribution(states, task.n_components)
        duration_stats = _duration_stats(durations)
        parameter_count = _model_parameter_count(task.n_components, n_features)

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
                "state_duration_distribution": json.dumps(durations),
                "state_duration_stats": json.dumps(duration_stats),
                "state_mean_duration": json.dumps(
                    [state_stats["mean"] for state_stats in duration_stats]
                ),
                "transition_matrix": json.dumps(model.transmat_.tolist()),
                "state_means": json.dumps(model.means_.tolist()),
                "state_covariances": json.dumps(model.covars_.tolist()),
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
    return _write_s3_parquet(
        s3,
        bucket,
        result["result_key"],
        pd.DataFrame([result]),
        metadata={
            "model-group": str(result["model_group"]),
            "success": str(bool(result["success"])).lower(),
            "grid-kind": "refined-full",
        },
    )


def _combine_results(s3: Any, bucket: str, output_prefix: str, run_id: str) -> pd.DataFrame:
    model_prefix = f"{output_prefix.strip('/')}/runs/run_id={run_id}/models"
    keys = [
        key
        for key in _list_s3_keys(s3, bucket, model_prefix)
        if key.endswith("/result.parquet")
    ]
    if not keys:
        raise FileNotFoundError(f"No result parquet files found under s3://{bucket}/{model_prefix}")

    frames = [_read_s3_parquet(s3, bucket, key) for key in sorted(keys)]
    combined = pd.concat(frames, ignore_index=True)
    sort_columns = ["model_group", "n_components", "random_state"]
    combined = combined.sort_values(sort_columns).reset_index(drop=True)
    _write_s3_parquet(
        s3,
        bucket,
        _refined_final_key(output_prefix, run_id),
        combined,
        metadata={
            "run-id": run_id,
            "rows": str(len(combined)),
            "successes": str(int(combined["success"].sum())),
            "failures": str(int((~combined["success"]).sum())),
            "grid-kind": "refined-full",
        },
    )
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run refined full-covariance GaussianHMM grid using top random states"
    )
    parser.add_argument("--bucket", default=os.getenv("YC_BUCKET", S3_BUCKET))
    parser.add_argument("--input-key", default=INPUT_KEY)
    parser.add_argument("--broad-output-prefix", default=BROAD_OUTPUT_PREFIX)
    parser.add_argument("--broad-run-id", help="Completed broad grid run_id; defaults to latest")
    parser.add_argument("--output-prefix", default=REFINED_OUTPUT_PREFIX)
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("refined_full_%Y%m%d_%H%M%S"),
        help="Stable id for resume; use the same value after a crash",
    )
    parser.add_argument("--top-random-states", type=int, default=TOP_RANDOM_STATES)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Explicit worker count; default is min(cpu_count - reserve, cap)",
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-final-combine", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker_cap = None if args.worker_cap == 0 else args.worker_cap
    max_workers = _resolve_max_workers(args.max_workers, args.worker_reserve, worker_cap)

    s3 = make_s3_client()
    broad_run_id = args.broad_run_id or _latest_run_id(
        s3,
        args.bucket,
        args.broad_output_prefix,
    )
    broad_key = _broad_all_results_key(args.broad_output_prefix, broad_run_id)
    print(f"Loading broad results: s3://{args.bucket}/{broad_key}", flush=True)
    broad_results = _read_s3_parquet(s3, args.bucket, broad_key)
    selected_states = _select_top_random_states(
        broad_results,
        top_n=args.top_random_states,
    )
    print("Selected random_state values from broad full-covariance BIC ranking:")
    for group, states in selected_states.items():
        print(
            f"  {group}: "
            + ", ".join(
                f"{item['random_state']} (broad_n={item['broad_n_components']}, "
                f"bic={item['broad_bic']:.2f})"
                for item in states
            ),
            flush=True,
        )

    print(f"Loading train dataset: s3://{args.bucket}/{args.input_key}", flush=True)
    dataset = _read_s3_parquet(s3, args.bucket, args.input_key)
    _validate_columns(dataset)
    print(f"Loaded rows={len(dataset):,} columns={len(dataset.columns):,}", flush=True)

    run_root = f"{args.output_prefix.strip('/')}/runs/run_id={args.run_id}"
    print(f"Run output: s3://{args.bucket}/{run_root}/", flush=True)

    with tempfile.TemporaryDirectory(prefix="hmm_refined_full_") as temp_dir:
        array_paths = _prepare_group_arrays(dataset, Path(temp_dir))
        del dataset
        gc.collect()

        all_tasks = _build_tasks(args.output_prefix, args.run_id, selected_states)
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
                    f"seed={task.random_state} seconds={result['train_seconds']} "
                    f"size_kib={size / 1024:.1f}",
                    flush=True,
                )

    if not args.skip_final_combine:
        combined = _combine_results(s3, args.bucket, args.output_prefix, args.run_id)
        print(
            f"Final table: s3://{args.bucket}/{_refined_final_key(args.output_prefix, args.run_id)} "
            f"rows={len(combined)} successes={int(combined['success'].sum())} "
            f"failures={int((~combined['success']).sum())}",
            flush=True,
        )
    print(f"Completed refined run_id={args.run_id}")


if __name__ == "__main__":
    main()
