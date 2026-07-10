from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.preprocessing import OrdinalEncoder


LOWER_IS_BETTER = {
    "RMSE",
    "MAE",
    "PinballLoss_05",
    "PinballLoss_25",
    "PinballLoss_50",
    "PinballLoss_75",
    "PinballLoss_95",
    "NLL",
    "IntervalWidth90",
    "IntervalWidth95",
    "train_time",
    "predict_time",
    "model_size_mb",
}


@dataclass(frozen=True)
class Stage1LoadResult:
    results: pd.DataFrame
    leaderboard: pd.DataFrame
    run_config: dict[str, Any]
    active_base_prefix: str
    active_s3_uri: str
    loaded_from_final_table: bool
    metric_keys: list[str]


def s3_exists(s3: Any, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True


def read_bytes(s3: Any, bucket: str, key: str) -> bytes:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def read_json(s3: Any, bucket: str, key: str) -> dict[str, Any]:
    return json.loads(read_bytes(s3, bucket, key).decode("utf-8"))


def read_parquet(s3: Any, bucket: str, key: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(read_bytes(s3, bucket, key)))


def list_keys(s3: Any, bucket: str, prefix: str, suffix: str | None = None) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if suffix is None or key.endswith(suffix):
                keys.append(key)
    return sorted(keys)


def resolve_latest_prefix(
    *,
    s3: Any,
    bucket: str,
    dataset_prefix: str,
    results_subdir: str,
    run_id: str,
) -> tuple[str, dict[str, Any]]:
    base_prefix = f"{dataset_prefix.strip('/')}/{results_subdir.strip('/')}/{run_id}"
    config_key = f"{base_prefix}/run_config.json"
    run_config = read_json(s3, bucket, config_key) if s3_exists(s3, bucket, config_key) else {}

    if run_id == "latest" and run_config.get("output_prefix"):
        expected = f"s3://{bucket}/"
        output_prefix = str(run_config["output_prefix"])
        if output_prefix.startswith(expected):
            base_prefix = output_prefix[len(expected) :].strip("/")
    return base_prefix, run_config


def load_stage1_results(
    *,
    s3: Any,
    bucket: str,
    dataset_prefix: str,
    results_subdir: str,
    run_id: str = "latest",
) -> Stage1LoadResult:
    active_base_prefix, run_config = resolve_latest_prefix(
        s3=s3,
        bucket=bucket,
        dataset_prefix=dataset_prefix,
        results_subdir=results_subdir,
        run_id=run_id,
    )
    active_s3_uri = f"s3://{bucket}/{active_base_prefix}/"
    results_key = f"{active_base_prefix}/stage1_results.parquet"
    leaderboard_key = f"{active_base_prefix}/leaderboard_top10.parquet"

    metric_keys: list[str] = []
    loaded_from_final_table = s3_exists(s3, bucket, results_key)
    if loaded_from_final_table:
        results = read_parquet(s3, bucket, results_key)
    else:
        metric_keys = list_keys(s3, bucket, f"{active_base_prefix}/jobs/", suffix="/metrics.json")
        results = pd.DataFrame(read_json(s3, bucket, key) for key in metric_keys)

    leaderboard = (
        read_parquet(s3, bucket, leaderboard_key)
        if s3_exists(s3, bucket, leaderboard_key)
        else pd.DataFrame()
    )
    return Stage1LoadResult(
        results=normalize_results(results),
        leaderboard=leaderboard,
        run_config=run_config,
        active_base_prefix=active_base_prefix,
        active_s3_uri=active_s3_uri,
        loaded_from_final_table=loaded_from_final_table,
        metric_keys=metric_keys,
    )


def normalize_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    frame = results.copy()
    numeric_columns = [
        "MAE",
        "RMSE",
        "R2",
        "Direction_Accuracy",
        "Direction_Accuracy_0.25%",
        "Pearson",
        "Spearman",
        "train_time",
        "predict_time",
        "model_size_mb",
        "alpha",
        "PinballLoss_05",
        "PinballLoss_25",
        "PinballLoss_50",
        "PinballLoss_75",
        "PinballLoss_95",
        "Coverage90",
        "Coverage95",
        "IntervalWidth90",
        "IntervalWidth95",
        "NLL",
        "CRPS",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "params" in frame.columns:
        param_frame = pd.json_normalize(frame["params"].map(_safe_json_loads))
        param_frame = param_frame.add_prefix("param_")
        frame = pd.concat([frame.reset_index(drop=True), param_frame.reset_index(drop=True)], axis=1)
    if "created_at_utc" in frame.columns:
        frame["created_at_utc"] = pd.to_datetime(frame["created_at_utc"], errors="coerce", utc=True)
    frame["family_loss"] = frame["model_family"].astype(str) + " / " + frame["loss_function"].astype(str)
    return frame


def _safe_json_loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def run_overview(results: pd.DataFrame, run_config: dict[str, Any]) -> pd.DataFrame:
    expected = int(run_config.get("jobs", 0) or 0)
    row = {
        "expected_jobs": expected,
        "completed_jobs": len(results),
        "coverage": len(results) / expected if expected else np.nan,
        "families": results["model_family"].nunique() if "model_family" in results else 0,
        "loss_functions": results["loss_function"].nunique() if "loss_function" in results else 0,
        "total_train_time_hours": results.get("train_time", pd.Series(dtype=float)).sum() / 3600,
        "median_train_time_sec": results.get("train_time", pd.Series(dtype=float)).median(),
        "best_RMSE": results.get("RMSE", pd.Series(dtype=float)).min(),
        "median_RMSE": results.get("RMSE", pd.Series(dtype=float)).median(),
    }
    return pd.DataFrame([row])


def add_composite_score(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    specs = [
        ("RMSE", True, 0.40),
        ("MAE", True, 0.25),
        ("Direction_Accuracy_0.25%", False, 0.20),
        ("Pearson", False, 0.075),
        ("Spearman", False, 0.075),
    ]
    score = np.zeros(len(frame), dtype=float)
    used_weight = 0.0
    for column, lower_is_better, weight in specs:
        if column not in frame.columns or frame[column].notna().sum() < 2:
            continue
        rank = frame[column].rank(ascending=lower_is_better, method="average", na_option="bottom")
        normalized = 1.0 - (rank - 1.0) / max(len(frame) - 1.0, 1.0)
        score += weight * normalized.to_numpy(dtype=float)
        used_weight += weight
    frame["CompositeScore"] = score / used_weight if used_weight else np.nan
    return frame


def top_tables(results: pd.DataFrame, n: int = 10) -> dict[str, pd.DataFrame]:
    frame = add_composite_score(results)
    columns = [
        "job_id",
        "model_family",
        "loss_function",
        "RMSE",
        "MAE",
        "Direction_Accuracy_0.25%",
        "CompositeScore",
        "train_time",
        "model_size_mb",
        "params",
    ]
    columns = [column for column in columns if column in frame.columns]
    tables = {
        "Top RMSE": frame.sort_values(["RMSE", "MAE", "job_id"]).loc[:, columns].head(n),
        "Top MAE": frame.sort_values(["MAE", "RMSE", "job_id"]).loc[:, columns].head(n),
        "Top Direction Accuracy": frame.sort_values(
            ["Direction_Accuracy_0.25%", "RMSE", "job_id"],
            ascending=[False, True, True],
        )
        .loc[:, columns]
        .head(n),
        "Top Composite Score": frame.sort_values(
            ["CompositeScore", "RMSE", "job_id"],
            ascending=[False, True, True],
        )
        .loc[:, columns]
        .head(n),
    }
    return tables


def parameter_columns(results: pd.DataFrame) -> list[str]:
    return sorted(column for column in results.columns if column.startswith("param_"))


def parameter_summary(
    results: pd.DataFrame,
    param_column: str,
    *,
    metric: str = "RMSE",
    top_metric: str = "RMSE",
    top_n: int = 10,
) -> pd.DataFrame:
    if param_column not in results.columns:
        return pd.DataFrame()
    ascending = top_metric in LOWER_IS_BETTER
    top_values = results.sort_values([top_metric, "job_id"], ascending=[ascending, True]).head(top_n)
    top_counts = top_values[param_column].value_counts(dropna=False).rename("top10_count")
    grouped = (
        results.groupby(param_column, dropna=False)[metric]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .rename(columns={"min": "best", "max": "worst"})
    )
    grouped = grouped.join(top_counts, how="left").fillna({"top10_count": 0})
    grouped["top10_count"] = grouped["top10_count"].astype(int)
    return grouped.reset_index().sort_values(["median", "best"], na_position="last")


def pair_pivot(
    results: pd.DataFrame,
    row_param: str,
    col_param: str,
    *,
    metric: str = "RMSE",
    aggfunc: str = "median",
) -> pd.DataFrame:
    return results.pivot_table(
        index=row_param,
        columns=col_param,
        values=metric,
        aggfunc=aggfunc,
        dropna=False,
    )


def plot_metric_distributions(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hist(results["train_time"].dropna(), bins=40)
    axes[0].set_title("Train time distribution")
    axes[0].set_xlabel("seconds")
    axes[1].hist(results["RMSE"].dropna(), bins=40)
    axes[1].set_title("RMSE distribution")
    axes[1].set_xlabel("RMSE")
    if "family_loss" in results.columns:
        results.boxplot(column="RMSE", by="family_loss", ax=axes[2], rot=45)
        axes[2].set_title("RMSE by family/loss")
        axes[2].set_xlabel("")
    fig.suptitle("")
    plt.tight_layout()


def plot_param_distribution(
    results: pd.DataFrame,
    param_column: str,
    *,
    metrics: tuple[str, ...] = ("RMSE", "MAE", "Direction_Accuracy_0.25%"),
) -> None:
    metrics = tuple(metric for metric in metrics if metric in results.columns)
    if not metrics or param_column not in results.columns:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        results.boxplot(column=metric, by=param_column, ax=ax, rot=45)
        ax.set_title(f"{metric} by {param_column.removeprefix('param_')}")
        ax.set_xlabel("")
    fig.suptitle("")
    plt.tight_layout()


def plot_violin_distribution(
    results: pd.DataFrame,
    param_column: str,
    *,
    metric: str = "RMSE",
) -> None:
    if param_column not in results.columns or metric not in results.columns:
        return
    grouped = [
        group[metric].dropna().to_numpy()
        for _, group in results.sort_values(param_column).groupby(param_column, dropna=False)
    ]
    labels = [
        str(value)
        for value in results.sort_values(param_column)[param_column].drop_duplicates().tolist()
    ]
    if not grouped:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.violinplot(grouped, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1), labels=labels, rotation=45)
    ax.set_title(f"{metric} distribution by {param_column.removeprefix('param_')}")
    ax.set_xlabel(param_column.removeprefix("param_"))
    ax.set_ylabel(metric)
    plt.tight_layout()


def plot_heatmap(pivot: pd.DataFrame, *, title: str) -> None:
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [str(column) for column in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)), [str(index) for index in pivot.index])
    ax.set_title(title)
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            if pd.notna(value):
                ax.text(col, row, f"{value:.6f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    plt.tight_layout()


def prepare_surrogate_matrix(results: pd.DataFrame, param_cols: list[str]) -> tuple[pd.DataFrame, OrdinalEncoder]:
    matrix = results.loc[:, param_cols].copy()
    for column in matrix.columns:
        matrix[column] = matrix[column].astype("string").fillna("__missing__")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoded = pd.DataFrame(
        encoder.fit_transform(matrix),
        columns=param_cols,
        index=matrix.index,
    )
    return encoded, encoder


def hyperparameter_importance(
    results: pd.DataFrame,
    *,
    metric: str = "RMSE",
    param_cols: list[str] | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, RandomForestRegressor | None, pd.DataFrame | None]:
    param_cols = param_cols or parameter_columns(results)
    usable = results.dropna(subset=[metric]).copy()
    param_cols = [column for column in param_cols if column in usable.columns and usable[column].nunique(dropna=False) > 1]
    if len(usable) < 8 or not param_cols:
        return pd.DataFrame(), None, None
    x, _ = prepare_surrogate_matrix(usable, param_cols)
    y = usable[metric].to_numpy(dtype=float)
    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x, y)
    permutation = permutation_importance(model, x, y, n_repeats=25, random_state=random_state, n_jobs=-1)
    importance = pd.DataFrame(
        {
            "parameter": param_cols,
            "rf_importance": model.feature_importances_,
            "permutation_importance_mean": permutation.importances_mean,
            "permutation_importance_std": permutation.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False)
    return importance, model, x


def plot_partial_dependence(
    model: RandomForestRegressor | None,
    x: pd.DataFrame | None,
    parameters: list[str],
) -> None:
    if model is None or x is None or not parameters:
        return
    selected = [parameter for parameter in parameters if parameter in x.columns]
    if not selected:
        return
    PartialDependenceDisplay.from_estimator(model, x, selected)
    plt.tight_layout()


def parameter_correlations(results: pd.DataFrame) -> pd.DataFrame:
    param_cols = parameter_columns(results)
    numeric_params = []
    for column in param_cols:
        numeric = pd.to_numeric(results[column], errors="coerce")
        if numeric.notna().sum() >= 3 and numeric.nunique(dropna=True) > 1:
            numeric_params.append(column)
    targets = [column for column in ["RMSE", "MAE", "Direction_Accuracy_0.25%", "train_time", "model_size_mb"] if column in results.columns]
    rows = []
    for param in numeric_params:
        for target in targets:
            valid = pd.to_numeric(results[param], errors="coerce").notna() & results[target].notna()
            if valid.sum() < 3:
                continue
            rows.append(
                {
                    "parameter": param,
                    "target": target,
                    "spearman": pd.to_numeric(results.loc[valid, param], errors="coerce").corr(
                        results.loc[valid, target],
                        method="spearman",
                    ),
                    "pearson": pd.to_numeric(results.loc[valid, param], errors="coerce").corr(
                        results.loc[valid, target],
                        method="pearson",
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)


def pareto_frontier(
    results: pd.DataFrame,
    *,
    quality_metric: str = "RMSE",
    cost_metric: str = "train_time",
) -> pd.DataFrame:
    frame = results.dropna(subset=[quality_metric, cost_metric]).sort_values([quality_metric, cost_metric]).copy()
    best_cost = math.inf
    keep = []
    for _, row in frame.iterrows():
        cost = float(row[cost_metric])
        if cost < best_cost:
            keep.append(True)
            best_cost = cost
        else:
            keep.append(False)
    return frame.loc[keep].copy()


def plot_pareto(results: pd.DataFrame, frontier: pd.DataFrame, *, quality_metric: str = "RMSE", cost_metric: str = "train_time") -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(results[cost_metric], results[quality_metric], alpha=0.55, label="models")
    if not frontier.empty:
        ordered = frontier.sort_values(cost_metric)
        ax.plot(ordered[cost_metric], ordered[quality_metric], marker="o", color="red", label="Pareto frontier")
    ax.set_xlabel(cost_metric)
    ax.set_ylabel(quality_metric)
    ax.set_title(f"Pareto frontier: {quality_metric} vs {cost_metric}")
    ax.legend()
    plt.tight_layout()


def stability_table(results: pd.DataFrame, *, metric: str = "RMSE", sizes: tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    rows = []
    ordered = results.dropna(subset=[metric]).sort_values([metric, "job_id"])
    best = ordered[metric].iloc[0] if len(ordered) else np.nan
    for size in sizes:
        subset = ordered.head(size)
        rows.append(
            {
                "top_n": size,
                "rows": len(subset),
                "best": subset[metric].min(),
                "worst": subset[metric].max(),
                "spread": subset[metric].max() - subset[metric].min(),
                "spread_vs_best_pct": ((subset[metric].max() - best) / abs(best) * 100) if pd.notna(best) and best else np.nan,
                "std": subset[metric].std(),
            }
        )
    return pd.DataFrame(rows)


def early_stop_curve(results: pd.DataFrame, *, metric: str = "RMSE") -> pd.DataFrame:
    if "created_at_utc" in results.columns and results["created_at_utc"].notna().any():
        ordered = results.sort_values(["created_at_utc", "job_id"]).copy()
    else:
        ordered = results.sort_values("job_id").copy()
    lower = metric in LOWER_IS_BETTER
    if lower:
        ordered["best_so_far"] = ordered[metric].cummin()
    else:
        ordered["best_so_far"] = ordered[metric].cummax()
    ordered["models_seen"] = np.arange(1, len(ordered) + 1)
    return ordered[["models_seen", "job_id", metric, "best_so_far"]]


def plot_early_stop(curve: pd.DataFrame, *, metric: str = "RMSE") -> None:
    if curve.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve["models_seen"], curve["best_so_far"], marker="o", markersize=3)
    ax.set_xlabel("models evaluated")
    ax.set_ylabel(f"best {metric} so far")
    ax.set_title("Early-stop diagnostic")
    plt.tight_layout()


def suggest_stage2_grid(
    results: pd.DataFrame,
    *,
    model_family: str = "catboost",
    loss_function: str | None = None,
    metric: str = "RMSE",
    top_n: int = 10,
) -> tuple[dict[str, list[Any]], list[str], pd.DataFrame]:
    subset = results.loc[results["model_family"].eq(model_family)].copy()
    if loss_function is not None:
        subset = subset.loc[subset["loss_function"].eq(loss_function)].copy()
    if subset.empty:
        return {}, ["No matching rows for selected model family/loss."], subset
    ascending = metric in LOWER_IS_BETTER
    top = subset.sort_values([metric, "job_id"], ascending=[ascending, True]).head(top_n)
    grid: dict[str, list[Any]] = {}
    notes: list[str] = []
    param_cols = parameter_columns(subset)
    for column in param_cols:
        name = column.removeprefix("param_")
        all_values = sorted(subset[column].dropna().unique().tolist())
        top_values = sorted(top[column].dropna().unique().tolist())
        if not top_values:
            continue
        grid[name] = _expand_stage2_values(name, top_values)
        excluded = [value for value in all_values if value not in top_values]
        if excluded:
            notes.append(f"{name}: exclude {excluded}; top-{top_n} used {top_values}.")
        else:
            notes.append(f"{name}: keep around {top_values}.")
    if model_family == "catboost":
        grid.setdefault("bagging_temperature", [0, 1, 3])
        grid.setdefault("rsm", [0.7, 0.85, 1.0])
        grid.setdefault("random_strength", [0, 1, 2])
    elif model_family == "lightgbm":
        grid.setdefault("feature_fraction", [0.6, 0.8, 1.0])
        grid.setdefault("bagging_fraction", [0.6, 0.8, 1.0])
        grid.setdefault("min_data_in_leaf", [20, 50, 100])
    return grid, notes, top


def _expand_stage2_values(name: str, values: list[Any]) -> list[Any]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.notna().all():
        vals = sorted(float(value) for value in numeric.tolist())
        if name in {"learning_rate"}:
            expanded = set()
            for value in vals:
                expanded.update([value * 0.7, value, value * 1.3])
            return sorted(round(value, 5) for value in expanded if value > 0)
        if name in {"l2_leaf_reg", "reg_lambda", "lambda_l2"}:
            expanded = set()
            for value in vals:
                expanded.update([value * 0.5, value, value * 1.5])
            return sorted(round(value, 5) for value in expanded if value > 0)
        if name in {"depth", "max_depth"}:
            expanded = set()
            for value in vals:
                center = int(round(value))
                expanded.update([center - 1, center, center + 1])
            return sorted(value for value in expanded if value > 0 or name == "max_depth")
        if name in {"num_leaves"}:
            expanded = set()
            for value in vals:
                center = int(round(value))
                expanded.update([max(2, center // 2), center, center * 2])
            return sorted(expanded)
        if len(set(vals)) == 1:
            only = vals[0]
            return [int(only) if float(only).is_integer() else only]
        return [int(value) if float(value).is_integer() else value for value in vals]
    return values


def format_grid_as_python(grid: dict[str, list[Any]], variable_name: str = "GRID_STAGE2") -> str:
    return variable_name + " = " + json.dumps(grid, ensure_ascii=False, indent=4)
