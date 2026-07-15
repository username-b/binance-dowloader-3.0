from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_SPECS = {
    10: ("dataset_target_10", "target_log_return_10m"),
    20: ("dataset_target_20", "target_log_return_20m"),
    30: ("dataset_target_30", "target_log_return_30m"),
}
DEFAULT_BUCKET = "binance-data-downloader"
DEFAULT_RESULTS_SUBDIR = "model_experiments"
DEFAULT_OUTPUT_SUBDIR = "stage3_hmm_lift_eval"
DEFAULT_HMM_SECTION = "with_ada_hmm_12h_24h"
DEFAULT_HMM_PREFIX = "ada_hmm_12h_24h_prob_state_"
STAGE3_NAME = "stage_3_microstructure"


@dataclass(frozen=True)
class TargetedModelSpec:
    horizon: int
    criterion: str
    family_label: str
    n_features: int


TARGETED_STAGE3_MODELS = (
    TargetedModelSpec(10, "MAE", "lasso", 12),
    TargetedModelSpec(20, "MAE", "lasso", 29),
    TargetedModelSpec(30, "MAE", "ridge", 16),
    TargetedModelSpec(10, "RMSE", "ols", 45),
    TargetedModelSpec(20, "RMSE", "ols", 45),
    TargetedModelSpec(30, "RMSE", "ols", 36),
    TargetedModelSpec(10, "DA", "huber", 25),
    TargetedModelSpec(20, "DA", "huber", 33),
    TargetedModelSpec(30, "DA", "huber", 23),
)


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

from build_price_feature_day import load_s3_parquet, make_s3_client
from train_target_10_models import (  # noqa: E402
    DEFAULT_DIRECTION_THRESHOLD,
    MODEL_FAMILIES,
    evaluate_model,
    prepare_target,
    upload_bytes,
    upload_dataframe_csv,
    upload_dataframe_parquet,
)


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("stage3_hmm_lift_eval")


def read_s3_parquet_required(s3: Any, bucket: str, key: str) -> pd.DataFrame:
    frame = load_s3_parquet(s3, bucket, key)
    if frame is None:
        raise FileNotFoundError(f"s3://{bucket}/{key}")
    return frame


def family_matches(model_family: str, family_label: str) -> bool:
    model_family = str(model_family).lower()
    family_label = family_label.lower()
    if family_label == "lasso":
        return model_family.startswith("lasso_")
    if family_label == "ridge":
        return model_family.startswith("ridge_")
    return model_family == family_label


def criterion_sort(spec: TargetedModelSpec) -> tuple[list[str], list[bool]]:
    if spec.criterion == "MAE":
        return ["MAE", "RMSE", "n_features", "simplicity_rank", "model_id"], [True, True, True, True, True]
    if spec.criterion == "RMSE":
        return ["RMSE", "MAE", "n_features", "simplicity_rank", "model_id"], [True, True, True, True, True]
    if spec.criterion == "DA":
        return ["Direction_Accuracy_025", "RMSE", "n_features", "simplicity_rank", "model_id"], [
            False,
            True,
            True,
            True,
            True,
        ]
    raise ValueError(f"Unsupported criterion: {spec.criterion}")


def select_stage3_model(results: pd.DataFrame, spec: TargetedModelSpec) -> pd.Series:
    frame = results.loc[results["stage"].astype(str).eq(STAGE3_NAME)].copy()
    frame = frame.loc[frame["n_features"].astype(int).eq(spec.n_features)].copy()
    frame = frame.loc[
        frame["model_family"].map(lambda value: family_matches(str(value), spec.family_label))
    ].copy()
    if frame.empty:
        raise ValueError(
            "No matching stage_3 model for "
            f"horizon={spec.horizon} criterion={spec.criterion} "
            f"family={spec.family_label} n_features={spec.n_features}"
        )
    sort_columns, ascending = criterion_sort(spec)
    return frame.sort_values(sort_columns, ascending=ascending, na_position="last").iloc[0]


def model_family_by_name(name: str):
    matches = [family for family in MODEL_FAMILIES if family.name == name]
    if not matches:
        raise ValueError(f"Unknown model_family from baseline results: {name}")
    return matches[0]


def target_baseline_sse(test: pd.DataFrame, target_column: str) -> float:
    y_test = prepare_target(test, target_column).dropna().to_numpy(dtype=np.float64)
    return float(np.square(y_test).sum())


def add_nrmse(row: dict[str, Any]) -> dict[str, Any]:
    true_std = float(row.get("true_std", np.nan))
    row["NRMSE"] = float(row["RMSE"] / true_std) if true_std > 0 else np.nan
    return row


def metric_delta(augmented: dict[str, Any], baseline: dict[str, Any], metric: str) -> float:
    return float(augmented.get(metric, np.nan) - baseline.get(metric, np.nan))


def pct_change_lower_is_better(augmented: dict[str, Any], baseline: dict[str, Any], metric: str) -> float:
    base = float(baseline.get(metric, np.nan))
    if not np.isfinite(base) or base == 0:
        return np.nan
    return float((base - float(augmented.get(metric, np.nan))) / abs(base) * 100.0)


def find_hmm_columns(train: pd.DataFrame, test: pd.DataFrame, prefix: str) -> list[str]:
    common = set(train.columns).intersection(test.columns)
    columns = sorted(column for column in common if column.startswith(prefix))
    if not columns:
        raise ValueError(f"No HMM probability columns found with prefix {prefix!r}")
    return columns


def evaluate_spec(
    *,
    s3: Any,
    bucket: str,
    results_subdir: str,
    hmm_section: str,
    hmm_prefix: str,
    direction_threshold: float,
    spec: TargetedModelSpec,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset_prefix, target_column = TARGET_SPECS[spec.horizon]
    baseline_results_key = f"{dataset_prefix}/{results_subdir.strip('/')}/latest/experiment_results.parquet"
    baseline_results = read_s3_parquet_required(s3, bucket, baseline_results_key)
    selected = select_stage3_model(baseline_results, spec)

    hmm_dataset_prefix = f"{dataset_prefix}/{hmm_section.strip('/')}"
    train = read_s3_parquet_required(s3, bucket, f"{hmm_dataset_prefix}/train.parquet")
    test = read_s3_parquet_required(s3, bucket, f"{hmm_dataset_prefix}/test.parquet")
    hmm_columns = find_hmm_columns(train, test, hmm_prefix)
    base_features = json.loads(selected["features"])
    missing_features = [feature for feature in base_features if feature not in train.columns or feature not in test.columns]
    if missing_features:
        raise ValueError(f"Selected baseline features are missing from HMM dataset: {missing_features}")

    family = model_family_by_name(str(selected["model_family"]))
    baseline_sse = target_baseline_sse(test, target_column)
    base_row, _ = evaluate_model(
        train,
        test,
        model_id=f"{selected['model_id']}__refit_stage3",
        stage=STAGE3_NAME,
        model_family=family,
        model_name=f"{selected['model_name']}__refit_stage3",
        feature_set=str(selected["feature_set"]),
        features=base_features,
        previous_model=str(selected.get("previous_model") or ""),
        added_block="stage3_refit_on_hmm_dataset",
        direction_threshold=direction_threshold,
        baseline_sse=baseline_sse,
        baseline_metrics=None,
        target_column=target_column,
    )
    hmm_row, _ = evaluate_model(
        train,
        test,
        model_id=f"{selected['model_id']}__stage3_plus_hmm",
        stage=f"{STAGE3_NAME}+hmm",
        model_family=family,
        model_name=f"{selected['model_name']}__stage3_plus_hmm",
        feature_set=f"{selected['feature_set']}+{hmm_section}",
        features=[*base_features, *hmm_columns],
        previous_model=str(selected["model_id"]),
        added_block=hmm_section,
        direction_threshold=direction_threshold,
        baseline_sse=baseline_sse,
        baseline_metrics=base_row,
        target_column=target_column,
    )
    base_row = add_nrmse(base_row)
    hmm_row = add_nrmse(hmm_row)

    comparison = {
        "horizon": spec.horizon,
        "criterion": spec.criterion,
        "family_label": spec.family_label,
        "selected_model_id": selected["model_id"],
        "selected_model_family": selected["model_family"],
        "selected_feature_set": selected["feature_set"],
        "base_n_features": len(base_features),
        "hmm_probability_features": len(hmm_columns),
        "augmented_n_features": len(base_features) + len(hmm_columns),
        "target_column": target_column,
        "baseline_results_key": baseline_results_key,
        "hmm_dataset_prefix": hmm_dataset_prefix,
        "baseline_NRMSE": base_row["NRMSE"],
        "augmented_NRMSE": hmm_row["NRMSE"],
        "delta_NRMSE": metric_delta(hmm_row, base_row, "NRMSE"),
        "NRMSE_improvement_pct": pct_change_lower_is_better(hmm_row, base_row, "NRMSE"),
        "baseline_RMSE": base_row["RMSE"],
        "augmented_RMSE": hmm_row["RMSE"],
        "delta_RMSE": metric_delta(hmm_row, base_row, "RMSE"),
        "RMSE_improvement_pct": pct_change_lower_is_better(hmm_row, base_row, "RMSE"),
        "baseline_DA": base_row["Direction_Accuracy"],
        "augmented_DA": hmm_row["Direction_Accuracy"],
        "delta_DA": metric_delta(hmm_row, base_row, "Direction_Accuracy"),
        "baseline_DA_025": base_row["Direction_Accuracy_025"],
        "augmented_DA_025": hmm_row["Direction_Accuracy_025"],
        "delta_DA_025": metric_delta(hmm_row, base_row, "Direction_Accuracy_025"),
        "baseline_OOS_R2": base_row["OOS_R2"],
        "augmented_OOS_R2": hmm_row["OOS_R2"],
        "delta_OOS_R2": metric_delta(hmm_row, base_row, "OOS_R2"),
    }
    return comparison, base_row, hmm_row


def save_outputs(
    *,
    s3: Any,
    bucket: str,
    output_subdir: str,
    run_id: str,
    comparisons: pd.DataFrame,
    refit_results: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    output_prefix = f"{output_subdir.strip('/')}/{run_id}"
    latest_prefix = f"{output_subdir.strip('/')}/latest"
    for prefix in (output_prefix, latest_prefix):
        upload_dataframe_parquet(s3, bucket, f"{prefix}/stage3_hmm_lift_comparison.parquet", comparisons)
        upload_dataframe_csv(s3, bucket, f"{prefix}/stage3_hmm_lift_comparison.csv", comparisons)
        upload_dataframe_parquet(s3, bucket, f"{prefix}/stage3_hmm_refit_results.parquet", refit_results)
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
        description="Evaluate whether ADA 12h/24h HMM probabilities improve selected stage_3 statistical models."
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--results-subdir", default=DEFAULT_RESULTS_SUBDIR)
    parser.add_argument("--hmm-section", default=DEFAULT_HMM_SECTION)
    parser.add_argument("--hmm-prefix", default=DEFAULT_HMM_PREFIX)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--direction-threshold", type=float, default=DEFAULT_DIRECTION_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    s3 = make_s3_client()
    comparisons = []
    refit_rows = []

    for spec in TARGETED_STAGE3_MODELS:
        logger.info(
            "START horizon=%dm criterion=%s family=%s n_features=%d",
            spec.horizon,
            spec.criterion,
            spec.family_label,
            spec.n_features,
        )
        comparison, base_row, hmm_row = evaluate_spec(
            s3=s3,
            bucket=args.bucket,
            results_subdir=args.results_subdir,
            hmm_section=args.hmm_section,
            hmm_prefix=args.hmm_prefix,
            direction_threshold=args.direction_threshold,
            spec=spec,
        )
        comparisons.append(comparison)
        refit_rows.extend([base_row, hmm_row])
        logger.info(
            "DONE horizon=%dm %s %s NRMSE %.6f -> %.6f delta=%+.6f DA %.6f -> %.6f delta=%+.6f",
            spec.horizon,
            spec.criterion,
            comparison["selected_model_family"],
            comparison["baseline_NRMSE"],
            comparison["augmented_NRMSE"],
            comparison["delta_NRMSE"],
            comparison["baseline_DA"],
            comparison["augmented_DA"],
            comparison["delta_DA"],
        )

    comparison_frame = pd.DataFrame(comparisons).sort_values(["horizon", "criterion", "family_label"])
    refit_frame = pd.DataFrame(refit_rows)
    print(comparison_frame.to_string(index=False))

    config = {
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": args.bucket,
        "results_subdir": args.results_subdir,
        "hmm_section": args.hmm_section,
        "hmm_prefix": args.hmm_prefix,
        "output_subdir": args.output_subdir,
        "direction_threshold": args.direction_threshold,
        "stage": STAGE3_NAME,
        "targeted_models": [spec.__dict__ for spec in TARGETED_STAGE3_MODELS],
    }
    if args.dry_run:
        logger.info("dry-run enabled; skipping upload")
        return
    output_uri = save_outputs(
        s3=s3,
        bucket=args.bucket,
        output_subdir=args.output_subdir,
        run_id=args.run_id,
        comparisons=comparison_frame,
        refit_results=refit_frame,
        config=config,
    )
    logger.info("uploaded=%s", output_uri)


if __name__ == "__main__":
    main()
