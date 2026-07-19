from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BUCKET = "binance-data-downloader"
DEFAULT_FAST_SUBDIR = "stage1_model_benchmarks/fast"
DEFAULT_STAGE3_GARCH_SUBDIR = "stage3_point_garch_probabilistic_eval"
DEFAULT_OUTPUT = "analysis/stage1_model_search/final_ml_vs_stat_garch_comparison.csv"


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "build_price_feature_day.py").exists():
            return candidate
    raise FileNotFoundError("Could not find project root with build_price_feature_day.py")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_price_feature_day import load_s3_parquet, make_s3_client  # noqa: E402


def read_s3_parquet_required(s3: Any, bucket: str, key: str) -> pd.DataFrame:
    frame = load_s3_parquet(s3, bucket, key)
    if frame is None:
        raise FileNotFoundError(f"s3://{bucket}/{key}")
    return frame


def read_s3_csv_required(s3: Any, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def nrmse_zero_from_oos_r2(value: float) -> float:
    return float(np.sqrt(max(0.0, 1.0 - float(value))))


def point_model_id(row: pd.Series) -> str:
    criterion = str(row["criterion"]).lower()
    family = str(row["family_label"]).lower()
    n_features = int(row["selected_n_features"])
    horizon = int(row["horizon"])
    return f"h{horizon}_{criterion}_{family}_{n_features}"


def normalize_point_results(point_results: pd.DataFrame) -> pd.DataFrame:
    frame = point_results.copy()
    frame["point_model"] = frame.apply(point_model_id, axis=1)
    frame["NRMSE_zero"] = frame["OOS_R2"].map(nrmse_zero_from_oos_r2)
    rename_map = {
        "Direction_Accuracy_025": "DA_025",
        "selected_n_features": "n_features",
    }
    frame = frame.rename(columns=rename_map)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    keep = [
        "horizon",
        "point_model",
        "criterion",
        "family_label",
        "n_features",
        "MAE",
        "RMSE",
        "NRMSE_zero",
        "DA_025",
        "OOS_R2",
    ]
    return frame[[col for col in keep if col in frame.columns]]


def build_ml_rows(s3: Any, bucket: str, fast_subdir: str, horizons: list[int]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for horizon in horizons:
        key = f"{fast_subdir.strip('/')}/horizon_{horizon}/latest/stage1_results.parquet"
        metrics = read_s3_parquet_required(s3, bucket, key)
        metrics["horizon"] = horizon
        rows.append(metrics)

    metrics = pd.concat(rows, ignore_index=True)
    metrics["model_group"] = "Fast ML"
    metrics["point_model"] = metrics["job_id"]
    metrics["probabilistic_model"] = metrics["job_id"]
    metrics["garch_model"] = ""
    metrics["distribution"] = metrics["loss_function"].fillna("")
    metrics["n_features"] = np.nan
    metrics["CRPS"] = metrics.get("CRPS", np.nan)
    metrics["Coverage90"] = metrics.get("Coverage90", np.nan)
    metrics["IntervalWidth90"] = metrics.get("IntervalWidth90", np.nan)
    metrics["DA_025"] = metrics.get("Direction_Accuracy_0.25%", np.nan)
    metrics["NRMSE_zero"] = metrics["OOS_R2"].map(nrmse_zero_from_oos_r2)
    metrics["quality_note"] = np.where(metrics["CRPS"].notna(), "probabilistic", "point_only")

    keep = [
        "model_group",
        "horizon",
        "point_model",
        "probabilistic_model",
        "garch_model",
        "distribution",
        "n_features",
        "MAE",
        "RMSE",
        "NRMSE_zero",
        "DA_025",
        "OOS_R2",
        "NLL",
        "CRPS",
        "Coverage90",
        "IntervalWidth90",
        "quality_note",
    ]
    return metrics[[col for col in keep if col in metrics.columns]]


def quality_note(row: pd.Series) -> str:
    if not np.isfinite(row.get("NLL", np.nan)) or not np.isfinite(row.get("CRPS", np.nan)):
        return "invalid_metrics"
    if row.get("IntervalWidth90", 0.0) > 0.05 or row.get("CRPS", 0.0) > 0.01:
        return "pathological_interval"
    if row.get("Coverage90", 0.0) > 0.995:
        return "overcoverage"
    return "usable_raw"


def build_stat_garch_rows(s3: Any, bucket: str, stage3_garch_subdir: str) -> pd.DataFrame:
    base = f"{stage3_garch_subdir.strip('/')}/latest"
    point = read_s3_parquet_required(s3, bucket, f"{base}/stage3_point_refit_results.parquet")
    prob = read_s3_parquet_required(s3, bucket, f"{base}/stage3_garch_probabilistic_results.parquet")

    point_norm = normalize_point_results(point)
    prob = prob.rename(columns={"CRPS_approx": "CRPS", "IntervalWidth90": "IntervalWidth90"}).copy()
    merged = prob.merge(
        point_norm,
        on=["horizon", "point_model"],
        how="left",
        suffixes=("_prob", ""),
    )
    merged["model_group"] = "Stat point + GARCH"
    merged["probabilistic_model"] = merged["model_name"]
    merged["quality_note"] = merged.apply(quality_note, axis=1)

    for col in ["MAE", "RMSE"]:
        if col not in merged.columns and f"{col}_prob" in merged.columns:
            merged[col] = merged[f"{col}_prob"]

    keep = [
        "model_group",
        "horizon",
        "point_model",
        "probabilistic_model",
        "garch_model",
        "distribution",
        "criterion",
        "family_label",
        "n_features",
        "MAE",
        "RMSE",
        "NRMSE_zero",
        "DA_025",
        "OOS_R2",
        "NLL",
        "CRPS",
        "Coverage90",
        "IntervalWidth90",
        "sigma_mean",
        "nu",
        "quality_note",
    ]
    return merged[[col for col in keep if col in merged.columns]]


def best_rows(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.copy()
    valid["rank_penalty"] = np.where(valid["quality_note"].isin(["pathological_interval", "invalid_metrics"]), 1, 0)
    return (
        valid.sort_values(["model_group", "horizon", "rank_penalty", "NLL", "CRPS", "IntervalWidth90"], na_position="last")
        .groupby(["model_group", "horizon"], as_index=False, group_keys=False)
        .head(1)
        .drop(columns=["rank_penalty"], errors="ignore")
        .reset_index(drop=True)
    )


def best_by_metric(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    valid = frame[np.isfinite(pd.to_numeric(frame[metric], errors="coerce"))].copy()
    valid = valid[~valid["quality_note"].isin(["pathological_interval", "invalid_metrics"])].copy()
    valid["selection_metric"] = metric
    return (
        valid.sort_values(["model_group", "horizon", metric, "IntervalWidth90"], na_position="last")
        .groupby(["model_group", "horizon"], as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )


def publication_summary(frame: pd.DataFrame) -> pd.DataFrame:
    parts = [best_by_metric(frame, "NLL"), best_by_metric(frame, "CRPS")]
    summary = pd.concat(parts, ignore_index=True, sort=False)
    summary = summary.sort_values(["horizon", "model_group", "selection_metric"])
    keep = [
        "selection_metric",
        "model_group",
        "horizon",
        "point_model",
        "probabilistic_model",
        "garch_model",
        "distribution",
        "criterion",
        "family_label",
        "n_features",
        "RMSE",
        "NRMSE_zero",
        "DA_025",
        "OOS_R2",
        "NLL",
        "CRPS",
        "Coverage90",
        "IntervalWidth90",
        "quality_note",
    ]
    return summary[[col for col in keep if col in summary.columns]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a final comparable table for Fast ML and statistical point + GARCH probabilistic models."
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--fast-subdir", default=DEFAULT_FAST_SUBDIR)
    parser.add_argument("--stage3-garch-subdir", default=DEFAULT_STAGE3_GARCH_SUBDIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--horizons", nargs="+", type=int, default=[10, 20, 30])
    args = parser.parse_args()

    s3 = make_s3_client()
    ml = build_ml_rows(s3, args.bucket, args.fast_subdir, args.horizons)
    stat_garch = build_stat_garch_rows(s3, args.bucket, args.stage3_garch_subdir)
    stat_garch = stat_garch[stat_garch["horizon"].isin(args.horizons)].copy()

    combined = pd.concat([ml, stat_garch], ignore_index=True, sort=False)
    combined = combined.sort_values(["horizon", "model_group", "NLL", "CRPS"], na_position="last")

    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    best = best_rows(combined)
    best_output = output_path.with_name(output_path.stem + "_best_by_horizon.csv")
    best.to_csv(best_output, index=False)

    summary = publication_summary(combined)
    summary_output = output_path.with_name(output_path.stem + "_publication_summary.csv")
    summary.to_csv(summary_output, index=False)

    print(f"Wrote full table: {output_path}")
    print(f"Wrote best-by-horizon table: {best_output}")
    print(f"Wrote publication summary: {summary_output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
