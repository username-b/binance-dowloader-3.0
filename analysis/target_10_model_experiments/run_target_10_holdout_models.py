from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


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
    DEFAULT_DATASET_PREFIX,
    DEFAULT_DIRECTION_THRESHOLD,
    DEFAULT_RESULTS_SUBDIR,
    TARGET_COLUMN,
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
    return logging.getLogger("target_10_holdout_models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run staged hold-out model experiments for dataset_target_10."
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--dataset-prefix", default=DEFAULT_DATASET_PREFIX)
    parser.add_argument("--results-subdir", default=DEFAULT_RESULTS_SUBDIR)
    parser.add_argument("--target", default=TARGET_COLUMN)
    parser.add_argument("--direction-threshold", type=float, default=DEFAULT_DIRECTION_THRESHOLD)
    parser.add_argument("--max-selected-per-stage", type=int, default=6)
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--parallel-backend", default="threading", choices=["threading", "loky"])
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run training and ranking but do not upload outputs to S3.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    run_id = args.run_id or make_run_id()
    started = time.perf_counter()

    logger.info("project_root=%s", PROJECT_ROOT)
    logger.info("run_id=%s", run_id)
    logger.info(
        "source=s3://%s/%s train/test target=%s",
        args.bucket,
        args.dataset_prefix.strip("/"),
        args.target,
    )
    logger.info(
        "selection=max_selected_per_stage=%d n_jobs=%d backend=%s threshold=%.6f",
        args.max_selected_per_stage,
        args.n_jobs,
        args.parallel_backend,
        args.direction_threshold,
    )

    s3 = make_s3_client()
    logger.info("loading train/test from S3")
    train, test = load_target_data(s3, args.bucket, args.dataset_prefix)
    logger.info(
        "loaded train rows=%d columns=%d; test rows=%d columns=%d",
        len(train),
        len(train.columns),
        len(test),
        len(test.columns),
    )

    logger.info("running experiments")
    results, stage_selection = run_experiments(
        train,
        test,
        target_column=args.target,
        direction_threshold=args.direction_threshold,
        max_selected_per_stage=args.max_selected_per_stage,
        n_jobs=args.n_jobs,
        parallel_backend=args.parallel_backend,
    )
    logger.info("evaluated models=%d stages=%d", len(results), len(stage_selection))

    for row in stage_selection.to_dict("records"):
        logger.info(
            "%s candidates=%d selected=%d",
            row["stage"],
            row["candidates"],
            row["selected"],
        )

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
            "  %s MAE=%.8f RMSE=%.8f features=%d",
            row["model_id"],
            row["MAE"],
            row["RMSE"],
            row["n_features"],
        )
    logger.info("top B by Direction_Accuracy_025:")
    for row in top_b.to_dict("records"):
        logger.info(
            "  %s DA_025=%.6f features=%d",
            row["model_id"],
            row["Direction_Accuracy_025"],
            row["n_features"],
        )

    run_config = {
        "run_id": run_id,
        "bucket": args.bucket,
        "dataset_prefix": args.dataset_prefix,
        "results_subdir": args.results_subdir,
        "target": args.target,
        "direction_threshold": args.direction_threshold,
        "max_selected_per_stage": args.max_selected_per_stage,
        "n_jobs": args.n_jobs,
        "parallel_backend": args.parallel_backend,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_columns": list(train.columns),
        "test_columns": list(test.columns),
    }

    if args.dry_run:
        logger.info("dry-run enabled; skipping S3 upload")
    else:
        logger.info("uploading outputs to S3")
        locations = save_outputs_to_s3(
            s3,
            args.bucket,
            args.dataset_prefix,
            args.results_subdir,
            run_id,
            results,
            stage_selection,
            run_config,
        )
        logger.info("run_prefix=%s", locations["run_prefix"])
        logger.info("latest_prefix=%s", locations["latest_prefix"])

    logger.info("done elapsed_sec=%.1f", time.perf_counter() - started)


if __name__ == "__main__":
    main()
