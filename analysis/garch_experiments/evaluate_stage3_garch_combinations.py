from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BUCKET = "binance-data-downloader"
DEFAULT_RESULTS_SUBDIR = "model_experiments"
DEFAULT_GARCH_SUBDIR = "garch_experiments"
DEFAULT_OUTPUT_SUBDIR = "stage3_point_garch_probabilistic_eval"
DEFAULT_TIMESTAMP_COLUMN = "timestamp"


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "build_price_feature_day.py").exists():
            return candidate
    raise FileNotFoundError("Could not find project root with build_price_feature_day.py")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGET_MODEL_DIR = PROJECT_ROOT / "analysis" / "target_10_model_experiments"
if str(TARGET_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(TARGET_MODEL_DIR))

from build_price_feature_day import load_s3_parquet, make_s3_client
from analysis.garch_experiments.evaluate_point_garch_combinations import (
    build_point_garch_forecasts,
    evaluate_point_garch_combinations,
    parse_requested_models,
    read_parquet,
    upload_csv,
    upload_parquet,
)
from evaluate_stage3_hmm_lift import (
    STAGE3_NAME,
    TARGETED_STAGE3_MODELS,
    TARGET_SPECS,
    TargetedModelSpec,
    model_family_by_name,
    select_stage3_model,
    target_baseline_sse,
)
from train_target_10_models import DEFAULT_DIRECTION_THRESHOLD, evaluate_model, prepare_target, upload_bytes


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("stage3_point_garch_probabilistic_eval")


def read_s3_parquet_required(s3: Any, bucket: str, key: str) -> pd.DataFrame:
    frame = load_s3_parquet(s3, bucket, key)
    if frame is None:
        raise FileNotFoundError(f"s3://{bucket}/{key}")
    return frame


def target_dataset_prefix(horizon: int) -> str:
    return TARGET_SPECS[horizon][0]


def target_column(horizon: int) -> str:
    return TARGET_SPECS[horizon][1]


def experiment_results_key(*, horizon: int, results_subdir: str, run_id: str) -> str:
    return f"{target_dataset_prefix(horizon)}/{results_subdir.strip('/')}/{run_id.strip('/')}/experiment_results.parquet"


def garch_prefix(*, horizon: int, garch_subdir: str, run_id: str) -> str:
    return f"{target_dataset_prefix(horizon)}/{garch_subdir.strip('/')}/{run_id.strip('/')}"


def load_target_train_test(
    *,
    s3: Any,
    bucket: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = target_dataset_prefix(horizon)
    train = read_s3_parquet_required(s3, bucket, f"{prefix}/train.parquet")
    test = read_s3_parquet_required(s3, bucket, f"{prefix}/test.parquet")
    return train, test


def point_prediction_frame(
    *,
    test: pd.DataFrame,
    target: str,
    y_pred: np.ndarray,
    model_name: str,
    timestamp_column: str,
) -> pd.DataFrame:
    y_test = prepare_target(test, target)
    test_mask = y_test.notna()
    frame = pd.DataFrame(
        {
            "y_true": y_test.loc[test_mask].to_numpy(dtype=np.float64),
            "y_pred": np.asarray(y_pred, dtype=np.float64),
        }
    )
    if timestamp_column in test.columns:
        frame.insert(0, "timestamp", test.loc[test_mask, timestamp_column].to_numpy())
    frame["stage3_model"] = model_name
    return frame


def refit_stage3_point_model(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    baseline_results: pd.DataFrame,
    spec: TargetedModelSpec,
    direction_threshold: float,
    timestamp_column: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    target = target_column(spec.horizon)
    selected = select_stage3_model(baseline_results, spec)
    features = json.loads(selected["features"])
    missing = [feature for feature in features if feature not in train.columns or feature not in test.columns]
    if missing:
        raise ValueError(f"Selected stage3 features are missing from target dataset: {missing}")

    row, y_pred = evaluate_model(
        train,
        test,
        model_id=f"{selected['model_id']}__stage3_refit_for_garch",
        stage=STAGE3_NAME,
        model_family=model_family_by_name(str(selected["model_family"])),
        model_name=f"{selected['model_name']}__stage3_refit_for_garch",
        feature_set=str(selected["feature_set"]),
        features=features,
        previous_model=str(selected.get("previous_model") or ""),
        added_block="stage3_refit_for_garch",
        direction_threshold=direction_threshold,
        baseline_sse=target_baseline_sse(test, target),
        baseline_metrics=None,
        target_column=target,
    )
    row.update(
        {
            "horizon": spec.horizon,
            "criterion": spec.criterion,
            "family_label": spec.family_label,
            "selected_model_id": selected["model_id"],
            "selected_model_family": selected["model_family"],
            "selected_model_name": selected["model_name"],
            "selected_feature_set": selected["feature_set"],
            "selected_n_features": int(selected["n_features"]),
        }
    )
    point_name = f"h{spec.horizon}_{spec.criterion.lower()}_{spec.family_label}_{spec.n_features}"
    predictions = point_prediction_frame(
        test=test,
        target=target,
        y_pred=y_pred,
        model_name=point_name,
        timestamp_column=timestamp_column,
    )
    return row, predictions


def evaluate_horizon(
    *,
    s3: Any,
    bucket: str,
    horizon: int,
    specs: list[TargetedModelSpec],
    results_subdir: str,
    stage3_run_id: str,
    garch_subdir: str,
    garch_run_id: str,
    garch_models: list[str] | None,
    garch_selection: str,
    garch_metric: str,
    direction_threshold: float,
    timestamp_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test = load_target_train_test(s3=s3, bucket=bucket, horizon=horizon)
    baseline_results = read_s3_parquet_required(
        s3,
        bucket,
        experiment_results_key(horizon=horizon, results_subdir=results_subdir, run_id=stage3_run_id),
    )
    point_rows = []
    point_predictions = {}
    for spec in specs:
        row, predictions = refit_stage3_point_model(
            train=train,
            test=test,
            baseline_results=baseline_results,
            spec=spec,
            direction_threshold=direction_threshold,
            timestamp_column=timestamp_column,
        )
        point_rows.append(row)
        point_predictions[str(predictions["stage3_model"].iloc[0])] = predictions

    prefix = garch_prefix(horizon=horizon, garch_subdir=garch_subdir, run_id=garch_run_id)
    garch_results = read_parquet(s3, bucket, f"{prefix}/garch_results.parquet")
    garch_predictions = read_parquet(s3, bucket, f"{prefix}/garch_predictions.parquet")
    metrics = evaluate_point_garch_combinations(
        point_predictions=point_predictions,
        garch_results=garch_results,
        garch_predictions=garch_predictions,
        garch_model_names=garch_models,
        garch_selection=garch_selection,
        garch_metric=garch_metric,
    )
    forecasts = build_point_garch_forecasts(
        point_predictions=point_predictions,
        garch_results=garch_results,
        garch_predictions=garch_predictions,
        garch_model_names=garch_models,
        garch_selection=garch_selection,
        garch_metric=garch_metric,
    )
    for frame in (metrics, forecasts):
        if not frame.empty:
            frame.insert(0, "horizon", horizon)
    point_frame = pd.DataFrame(point_rows)
    return point_frame, metrics, forecasts


def save_outputs(
    *,
    s3: Any,
    bucket: str,
    output_subdir: str,
    run_id: str,
    point_results: pd.DataFrame,
    probabilistic_results: pd.DataFrame,
    probabilistic_predictions: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    output_prefix = f"{output_subdir.strip('/')}/{run_id.strip('/')}"
    latest_prefix = f"{output_subdir.strip('/')}/latest"
    for prefix in (output_prefix, latest_prefix):
        upload_parquet(s3, bucket, f"{prefix}/stage3_point_refit_results.parquet", point_results)
        upload_csv(s3, bucket, f"{prefix}/stage3_point_refit_results.csv", point_results)
        upload_parquet(
            s3,
            bucket,
            f"{prefix}/stage3_garch_probabilistic_results.parquet",
            probabilistic_results,
        )
        upload_csv(
            s3,
            bucket,
            f"{prefix}/stage3_garch_probabilistic_results.csv",
            probabilistic_results,
        )
        upload_parquet(
            s3,
            bucket,
            f"{prefix}/stage3_garch_probabilistic_predictions.parquet",
            probabilistic_predictions,
        )
        upload_bytes(
            s3,
            bucket,
            f"{prefix}/run_config.json",
            json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8"),
            "application/json",
        )
    return f"s3://{bucket}/{output_prefix}/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Student-t probabilistic forecasts from selected stage_3 statistical models plus GARCH volatility."
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--results-subdir", default=DEFAULT_RESULTS_SUBDIR)
    parser.add_argument("--stage3-run-id", default="latest")
    parser.add_argument("--garch-subdir", default=DEFAULT_GARCH_SUBDIR)
    parser.add_argument("--garch-run-id", default="latest")
    parser.add_argument("--garch-selection", default="all", choices=["all", "best"])
    parser.add_argument("--garch-metric", default="test_nll")
    parser.add_argument("--garch-model", action="append", default=None)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--direction-threshold", type=float, default=DEFAULT_DIRECTION_THRESHOLD)
    parser.add_argument("--timestamp-column", default=DEFAULT_TIMESTAMP_COLUMN)
    parser.add_argument("--horizon", type=int, choices=sorted(TARGET_SPECS), action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    s3 = make_s3_client()
    requested_horizons = set(args.horizon or sorted(TARGET_SPECS))
    garch_models = parse_requested_models(args.garch_model)

    point_frames = []
    result_frames = []
    prediction_frames = []
    for horizon in sorted(requested_horizons):
        specs = [spec for spec in TARGETED_STAGE3_MODELS if spec.horizon == horizon]
        logger.info("START horizon=%dm stage3_models=%d", horizon, len(specs))
        point_results, probabilistic_results, probabilistic_predictions = evaluate_horizon(
            s3=s3,
            bucket=args.bucket,
            horizon=horizon,
            specs=specs,
            results_subdir=args.results_subdir,
            stage3_run_id=args.stage3_run_id,
            garch_subdir=args.garch_subdir,
            garch_run_id=args.garch_run_id,
            garch_models=garch_models,
            garch_selection=args.garch_selection,
            garch_metric=args.garch_metric,
            direction_threshold=args.direction_threshold,
            timestamp_column=args.timestamp_column,
        )
        point_frames.append(point_results)
        result_frames.append(probabilistic_results)
        prediction_frames.append(probabilistic_predictions)
        logger.info(
            "DONE horizon=%dm point_models=%d probabilistic_rows=%d prediction_rows=%d",
            horizon,
            len(point_results),
            len(probabilistic_results),
            len(probabilistic_predictions),
        )

    point_results = pd.concat(point_frames, ignore_index=True)
    probabilistic_results = pd.concat(result_frames, ignore_index=True).sort_values(
        ["horizon", "NLL", "CRPS_approx", "model_name"],
        na_position="last",
    )
    probabilistic_predictions = pd.concat(prediction_frames, ignore_index=True)
    print(probabilistic_results.to_string(index=False))

    config = {
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": args.bucket,
        "results_subdir": args.results_subdir,
        "stage3_run_id": args.stage3_run_id,
        "garch_subdir": args.garch_subdir,
        "garch_run_id": args.garch_run_id,
        "garch_selection": args.garch_selection,
        "garch_metric": args.garch_metric,
        "garch_models": garch_models,
        "output_subdir": args.output_subdir,
        "direction_threshold": args.direction_threshold,
        "timestamp_column": args.timestamp_column,
        "horizons": sorted(requested_horizons),
        "targeted_models": [spec.__dict__ for spec in TARGETED_STAGE3_MODELS if spec.horizon in requested_horizons],
    }
    if args.dry_run:
        logger.info("dry-run enabled; skipping upload")
        return

    output_uri = save_outputs(
        s3=s3,
        bucket=args.bucket,
        output_subdir=args.output_subdir,
        run_id=args.run_id,
        point_results=point_results,
        probabilistic_results=probabilistic_results,
        probabilistic_predictions=probabilistic_predictions,
        config=config,
    )
    logger.info("uploaded=%s", output_uri)


if __name__ == "__main__":
    main()
