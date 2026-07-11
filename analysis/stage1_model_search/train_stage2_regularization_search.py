from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "build_price_feature_day.py").exists():
            return candidate
    raise FileNotFoundError("Could not find project root with build_price_feature_day.py")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.stage1_model_search.train_stage1_architecture_search import (
    DEFAULT_BUCKET,
    DEFAULT_DIRECTION_THRESHOLD,
    DEFAULT_HMM_FEATURE_PREFIX,
    TARGET_SPECS,
    SearchJob,
    build_leaderboards,
    collect_metrics,
    load_target_data,
    make_run_id,
    make_s3_client,
    object_exists,
    prepare_data,
    run_one_job,
    upload_csv,
    upload_json,
    upload_parquet,
)


DEFAULT_RESULTS_SUBDIR = "stage2_regularization_search"
DEFAULT_STAGE1_RUN_ID = "20260709_204850"


ARCHITECTURE_SEEDS_STAGE2_MAIN = [
    {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 10},
    {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 3},
    {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 30},
    {"depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 10},
    {"depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 30},
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3},
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 10},
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 30},
]


ARCHITECTURE_SEEDS_STAGE2_FAST = [
    {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 10},
    {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 30},
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 10},
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 30},
]


CATBOOST_REGULARIZATION_GRID = {
    "bagging_temperature": [0, 1, 3, 5],
    "rsm": [0.7, 0.85, 1.0],
    "random_strength": [0, 1, 2],
}


LIGHTGBM_ARCHITECTURE_SEEDS_STAGE2 = [
    {"num_leaves": 31, "learning_rate": 0.03, "max_depth": 10},
    {"num_leaves": 31, "learning_rate": 0.03, "max_depth": -1},
    {"num_leaves": 31, "learning_rate": 0.05, "max_depth": -1},
    {"num_leaves": 63, "learning_rate": 0.03, "max_depth": -1},
]


LIGHTGBM_REGULARIZATION_GRID = {
    "feature_fraction": [0.6, 0.8, 1.0],
    "bagging_fraction": [0.6, 0.8, 1.0],
    "min_data_in_leaf": [20, 50, 100],
}


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("stage2_regularization_search")


def regularization_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    combinations: list[dict[str, Any]] = [{}]
    for key in keys:
        combinations = [
            {**existing, key: value}
            for existing in combinations
            for value in grid[key]
        ]
    return combinations


def make_catboost_stage2_jobs(*, mode: str, loss_function: str) -> list[SearchJob]:
    seeds = ARCHITECTURE_SEEDS_STAGE2_FAST if mode == "fast" else ARCHITECTURE_SEEDS_STAGE2_MAIN
    regularization = regularization_combinations(CATBOOST_REGULARIZATION_GRID)
    jobs = []
    for seed_index, seed in enumerate(seeds, start=1):
        for reg_index, reg in enumerate(regularization, start=1):
            params = {
                **seed,
                **reg,
                "bootstrap_type": "Bayesian",
                "iterations": 3000,
            }
            job_id = (
                f"catboost_uncertainty_seed{seed_index:02d}_reg{reg_index:02d}_"
                f"d{seed['depth']}_lr{seed['learning_rate']}_l2{seed['l2_leaf_reg']}_"
                f"bt{reg['bagging_temperature']}_rsm{reg['rsm']}_rs{reg['random_strength']}"
            )
            jobs.append(SearchJob(job_id, "catboost", loss_function, params))
    return jobs


def make_lightgbm_stage2_jobs() -> list[SearchJob]:
    regularization = regularization_combinations(LIGHTGBM_REGULARIZATION_GRID)
    jobs = []
    for seed_index, seed in enumerate(LIGHTGBM_ARCHITECTURE_SEEDS_STAGE2, start=1):
        for reg_index, reg in enumerate(regularization, start=1):
            params = {
                **seed,
                **reg,
                "bagging_freq": 1,
                "n_estimators": 3000,
            }
            job_id = (
                f"lightgbm_rmse_seed{seed_index:02d}_reg{reg_index:02d}_"
                f"leaves{seed['num_leaves']}_lr{seed['learning_rate']}_depth{seed['max_depth']}_"
                f"ff{reg['feature_fraction']}_bf{reg['bagging_fraction']}_minleaf{reg['min_data_in_leaf']}"
            )
            jobs.append(SearchJob(job_id, "lightgbm", "RMSE", params))
    return jobs


def build_jobs(*, mode: str, include_lightgbm: bool, limit_jobs: int | None) -> list[SearchJob]:
    jobs = make_catboost_stage2_jobs(mode=mode, loss_function="RMSEWithUncertainty")
    if include_lightgbm:
        jobs.extend(make_lightgbm_stage2_jobs())
    if limit_jobs is not None:
        jobs = jobs[:limit_jobs]
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 regularization search.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--horizon", type=int, default=20, choices=sorted(TARGET_SPECS))
    parser.add_argument("--dataset-prefix", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--results-subdir", default=DEFAULT_RESULTS_SUBDIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stage1-run-id", default=DEFAULT_STAGE1_RUN_ID)
    parser.add_argument("--mode", choices=["fast", "main"], default="main")
    parser.add_argument("--include-lightgbm", action="store_true")
    parser.add_argument("--max-parallel-models", type=int, default=4)
    parser.add_argument("--threads-per-model", type=int, default=4)
    parser.add_argument("--direction-threshold", type=float, default=DEFAULT_DIRECTION_THRESHOLD)
    parser.add_argument("--hmm-feature-prefix", default=DEFAULT_HMM_FEATURE_PREFIX)
    parser.add_argument("--allow-missing-hmm", action="store_true")
    parser.add_argument("--limit-jobs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    run_id = args.run_id or make_run_id()
    default_dataset_prefix, default_target = TARGET_SPECS[args.horizon]
    dataset_prefix = args.dataset_prefix or default_dataset_prefix
    target_column = args.target or default_target
    output_prefix = f"{dataset_prefix.strip('/')}/{args.results_subdir.strip('/')}/{run_id}"
    latest_prefix = f"{dataset_prefix.strip('/')}/{args.results_subdir.strip('/')}/latest"
    jobs = build_jobs(
        mode=args.mode,
        include_lightgbm=args.include_lightgbm,
        limit_jobs=args.limit_jobs,
    )
    run_config = {
        "run_id": run_id,
        "stage": 2,
        "stage1_run_id": args.stage1_run_id,
        "stage1_source_prefix": (
            f"s3://{args.bucket}/{dataset_prefix.strip('/')}/"
            f"stage1_architecture_search/{args.stage1_run_id}/"
        ),
        "horizon": args.horizon,
        "dataset_prefix": dataset_prefix,
        "target_column": target_column,
        "results_subdir": args.results_subdir,
        "output_prefix": f"s3://{args.bucket}/{output_prefix}/",
        "mode": args.mode,
        "include_lightgbm": args.include_lightgbm,
        "max_parallel_models": args.max_parallel_models,
        "threads_per_model": args.threads_per_model,
        "reserved_vcpu_hint": max(0, 32 - args.max_parallel_models * args.threads_per_model),
        "direction_threshold": args.direction_threshold,
        "hmm_feature_prefix": args.hmm_feature_prefix,
        "require_hmm_features": not args.allow_missing_hmm,
        "architecture_seeds": (
            ARCHITECTURE_SEEDS_STAGE2_FAST
            if args.mode == "fast"
            else ARCHITECTURE_SEEDS_STAGE2_MAIN
        ),
        "catboost_regularization_grid": CATBOOST_REGULARIZATION_GRID,
        "lightgbm_architecture_seeds": LIGHTGBM_ARCHITECTURE_SEEDS_STAGE2 if args.include_lightgbm else [],
        "lightgbm_regularization_grid": LIGHTGBM_REGULARIZATION_GRID if args.include_lightgbm else {},
        "jobs": len(jobs),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    logger.info("run_prefix=s3://%s/%s/", args.bucket, output_prefix)
    logger.info(
        "jobs=%d mode=%s max_parallel_models=%d threads_per_model=%d",
        len(jobs),
        args.mode,
        args.max_parallel_models,
        args.threads_per_model,
    )
    if args.dry_run:
        counts = pd.DataFrame([job.__dict__ for job in jobs]).groupby(
            ["model_family", "loss_function"],
            dropna=False,
        ).size()
        for (model_family, loss_function), count in counts.items():
            logger.info("queue %s %s jobs=%d", model_family, loss_function, count)
        logger.info("dry-run enabled; skipping S3 upload and training")
        return

    s3 = make_s3_client()
    upload_json(s3, args.bucket, f"{output_prefix}/run_config.json", run_config)
    upload_json(s3, args.bucket, f"{latest_prefix}/run_config.json", run_config)
    upload_parquet(s3, args.bucket, f"{output_prefix}/job_queue.parquet", pd.DataFrame([job.__dict__ for job in jobs]))

    logger.info("loading train/test from s3://%s/%s", args.bucket, dataset_prefix)
    train, test = load_target_data(s3, args.bucket, dataset_prefix)
    data = prepare_data(
        train,
        test,
        target_column,
        hmm_feature_prefix=args.hmm_feature_prefix,
        require_hmm_features=not args.allow_missing_hmm,
    )
    logger.info(
        "prepared train_rows=%d test_rows=%d features=%d",
        len(data.y_train),
        len(data.y_test),
        len(data.feature_columns),
    )

    completed = 0
    skipped = 0
    failed = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.max_parallel_models) as executor:
        futures = {
            executor.submit(
                run_one_job,
                job=job,
                data=data,
                bucket=args.bucket,
                output_prefix=output_prefix,
                direction_threshold=args.direction_threshold,
                threads_per_model=args.threads_per_model,
                overwrite=args.overwrite,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                if result.get("status") == "skipped":
                    skipped += 1
                    logger.info("SKIP %s", job.job_id)
                else:
                    completed += 1
                    logger.info(
                        "DONE %s RMSE=%.8f DA_025=%.6f elapsed=%.1fs",
                        job.job_id,
                        result["RMSE"],
                        result["Direction_Accuracy_0.25%"],
                        result["elapsed_sec"],
                    )
            except Exception as exc:
                failed.append({"job_id": job.job_id, "error": str(exc)})
                logger.exception("FAILED %s", job.job_id)

    results = collect_metrics(s3, args.bucket, output_prefix)
    leaderboard = build_leaderboards(results, top_n=10)
    upload_parquet(s3, args.bucket, f"{output_prefix}/stage2_results.parquet", results)
    upload_csv(s3, args.bucket, f"{output_prefix}/stage2_results.csv", results)
    upload_parquet(s3, args.bucket, f"{output_prefix}/leaderboard_top10.parquet", leaderboard)
    upload_csv(s3, args.bucket, f"{output_prefix}/leaderboard_top10.csv", leaderboard)
    upload_parquet(s3, args.bucket, f"{latest_prefix}/stage2_results.parquet", results)
    upload_parquet(s3, args.bucket, f"{latest_prefix}/leaderboard_top10.parquet", leaderboard)
    upload_json(
        s3,
        args.bucket,
        f"{output_prefix}/run_summary.json",
        {
            **run_config,
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "elapsed_sec": time.perf_counter() - started,
            "result_rows": len(results),
        },
    )
    logger.info(
        "completed=%d skipped=%d failed=%d elapsed_sec=%.1f",
        completed,
        skipped,
        len(failed),
        time.perf_counter() - started,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
