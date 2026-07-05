from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


TARGET_SPECS = {
    10: {
        "dataset_prefix": "dataset_target_10",
        "target_column": "target_log_return_10m",
    },
    20: {
        "dataset_prefix": "dataset_target_20",
        "target_column": "target_log_return_20m",
    },
    30: {
        "dataset_prefix": "dataset_target_30",
        "target_column": "target_log_return_30m",
    },
}


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "build_price_feature_day.py").exists():
            return candidate
    raise FileNotFoundError("Could not find project root with build_price_feature_day.py")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODULE_DIR = PROJECT_ROOT / "analysis" / "target_10_model_experiments"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from build_price_feature_day import make_s3_client
from train_target_10_models import (
    DEFAULT_BUCKET,
    DEFAULT_DIRECTION_THRESHOLD,
    DEFAULT_RESULTS_SUBDIR,
    load_target_data,
    make_run_id,
    run_experiments,
    save_outputs_to_s3,
)


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger("target_holdout_models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run staged hold-out model experiments for target horizons 10/20/30."
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[20, 30],
        choices=sorted(TARGET_SPECS),
        help="Forecast horizons to run. Defaults to 20 30.",
    )
    parser.add_argument("--results-subdir", default=DEFAULT_RESULTS_SUBDIR)
    parser.add_argument("--direction-threshold", type=float, default=DEFAULT_DIRECTION_THRESHOLD)
    parser.add_argument("--max-selected-per-stage", type=int, default=6)
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--parallel-backend", default="threading", choices=["threading", "loky"])
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id shared by all selected horizons. Defaults to current UTC timestamp.",
    )
    parser.add_argument(
        "--no-model-progress",
        action="store_true",
        help="Disable per-model progress prints from the training loop.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run training and ranking but do not upload outputs to S3.",
    )
    return parser.parse_args()


def log_top_models(logger: logging.Logger, results) -> None:
    top_a = results.sort_values(
        ["MAE", "RMSE", "n_features", "simplicity_rank", "model_id"],
        ascending=[True, True, True, True, True],
    ).head(3)
    top_b = results.sort_values(
        ["Direction_Accuracy_025", "n_features", "simplicity_rank", "model_id"],
        ascending=[False, True, True, True],
        na_position="last",
    ).head(3)

    logger.info("top A by MAE/RMSE:")
    for row in top_a.to_dict("records"):
        logger.info(
            "  %s MAE=%.8f RMSE=%.8f DA_025=%.6f features=%d",
            row["model_id"],
            row["MAE"],
            row["RMSE"],
            row["Direction_Accuracy_025"],
            row["n_features"],
        )

    logger.info("top B by Direction_Accuracy_025:")
    for row in top_b.to_dict("records"):
        logger.info(
            "  %s DA_025=%.6f MAE=%.8f RMSE=%.8f features=%d",
            row["model_id"],
            row["Direction_Accuracy_025"],
            row["MAE"],
            row["RMSE"],
            row["n_features"],
        )


def run_one_horizon(
    *,
    horizon: int,
    bucket: str,
    results_subdir: str,
    direction_threshold: float,
    max_selected_per_stage: int,
    n_jobs: int,
    parallel_backend: str,
    run_id: str,
    model_progress: bool,
    dry_run: bool,
    s3,
    logger: logging.Logger,
) -> None:
    spec = TARGET_SPECS[horizon]
    dataset_prefix = spec["dataset_prefix"]
    target_column = spec["target_column"]
    started = time.perf_counter()

    logger.info(
        "START horizon=%dm source=s3://%s/%s target=%s",
        horizon,
        bucket,
        dataset_prefix,
        target_column,
    )
    train, test = load_target_data(s3, bucket, dataset_prefix)
    if target_column not in train.columns or target_column not in test.columns:
        raise ValueError(
            f"Target column {target_column!r} is missing in {dataset_prefix} train/test"
        )

    logger.info(
        "horizon=%dm loaded train rows=%d columns=%d; test rows=%d columns=%d",
        horizon,
        len(train),
        len(train.columns),
        len(test),
        len(test.columns),
    )

    results, stage_selection = run_experiments(
        train,
        test,
        target_column=target_column,
        direction_threshold=direction_threshold,
        max_selected_per_stage=max_selected_per_stage,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        print_progress=model_progress,
    )
    logger.info(
        "horizon=%dm evaluated models=%d stages=%d",
        horizon,
        len(results),
        len(stage_selection),
    )
    for row in stage_selection.to_dict("records"):
        logger.info(
            "horizon=%dm %s candidates=%d selected=%d prepared_features=%d",
            horizon,
            row["stage"],
            row["candidates"],
            row["selected"],
            row["prepared_features"],
        )
    log_top_models(logger, results)

    run_config = {
        "run_id": run_id,
        "horizon_minutes": horizon,
        "bucket": bucket,
        "dataset_prefix": dataset_prefix,
        "results_subdir": results_subdir,
        "target": target_column,
        "direction_threshold": direction_threshold,
        "max_selected_per_stage": max_selected_per_stage,
        "n_jobs": n_jobs,
        "parallel_backend": parallel_backend,
        "model_progress": model_progress,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_columns": list(train.columns),
        "test_columns": list(test.columns),
    }

    if dry_run:
        logger.info("horizon=%dm dry-run enabled; skipping S3 upload", horizon)
    else:
        locations = save_outputs_to_s3(
            s3,
            bucket,
            dataset_prefix,
            results_subdir,
            run_id,
            results,
            stage_selection,
            run_config,
        )
        logger.info("horizon=%dm run_prefix=%s", horizon, locations["run_prefix"])
        logger.info("horizon=%dm latest_prefix=%s", horizon, locations["latest_prefix"])

    logger.info("DONE horizon=%dm elapsed_sec=%.1f", horizon, time.perf_counter() - started)


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    run_id = args.run_id or make_run_id()
    started = time.perf_counter()

    logger.info("project_root=%s", PROJECT_ROOT)
    logger.info("run_id=%s", run_id)
    logger.info("horizons=%s", ",".join(str(horizon) for horizon in args.horizons))
    logger.info(
        "selection=max_selected_per_stage=%d n_jobs=%d backend=%s threshold=%.6f",
        args.max_selected_per_stage,
        args.n_jobs,
        args.parallel_backend,
        args.direction_threshold,
    )

    s3 = make_s3_client()
    for horizon in args.horizons:
        run_one_horizon(
            horizon=horizon,
            bucket=args.bucket,
            results_subdir=args.results_subdir,
            direction_threshold=args.direction_threshold,
            max_selected_per_stage=args.max_selected_per_stage,
            n_jobs=args.n_jobs,
            parallel_backend=args.parallel_backend,
            run_id=run_id,
            model_progress=not args.no_model_progress,
            dry_run=args.dry_run,
            s3=s3,
            logger=logger,
        )

    logger.info("ALL DONE elapsed_sec=%.1f", time.perf_counter() - started)


if __name__ == "__main__":
    main()
