"""Analyze feature informativeness for prepared feature datasets.

The script intentionally avoids predictive ML models. It reads a prepared
parquet dataset, computes descriptive statistics, target relationships,
target distribution by feature deciles, non-parametric distribution tests, and
feature-to-feature correlations, then writes compact parquet outputs.
"""

from __future__ import annotations

import argparse
import gc
import io
import logging
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from botocore.exceptions import ClientError

from build_price_feature_day import load_s3_parquet, make_s3_client, unified_dataset_key

try:
    from scipy import stats
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    stats = None


DEFAULT_CONFIG_PATH = "feature_informativeness_config.yaml"
PARQUET_CONTENT_TYPE = "application/vnd.apache.parquet"
SCHEMA_VERSION = "1"


@contextmanager
def timed_step(logger: logging.Logger, name: str):
    start = time.perf_counter()
    logger.info("start %s", name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("finish %s elapsed_sec=%.3f", name, elapsed)


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger("feature_informativeness")


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    return config


def _source_dates(source: dict[str, Any]) -> pd.DatetimeIndex:
    start = source.get("start_date")
    end = source.get("end_date")
    if not start or not end:
        raise ValueError("source.start_date and source.end_date are required")
    dates = pd.date_range(pd.Timestamp(start).date(), pd.Timestamp(end).date(), freq="D")
    if dates.empty:
        raise ValueError("Source date range is empty")
    return dates


def load_dataset(config: dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    source = config["source"]
    source_type = source.get("type", "s3_partitions")
    if source_type == "local_parquet":
        local_path = source.get("local_path")
        if not local_path:
            raise ValueError("source.local_path is required for local_parquet")
        logger.info("reading local parquet path=%s", local_path)
        return pd.read_parquet(local_path)

    if source_type == "s3_file":
        s3 = make_s3_client()
        bucket = source["bucket"]
        key = source["key"]
        logger.info("reading s3 parquet s3://%s/%s", bucket, key)
        frame = load_s3_parquet(s3, bucket, key)
        if frame is None:
            raise FileNotFoundError(f"S3 source file not found: s3://{bucket}/{key}")
        return frame

    if source_type != "s3_partitions":
        raise ValueError(f"Unsupported source.type: {source_type}")

    s3 = make_s3_client()
    bucket = source["bucket"]
    prefix = source.get("prefix", "features/unified_dataset")
    symbol = source.get("symbol", "ADAUSDT")
    interval = source.get("interval", "1m")
    frames: list[pd.DataFrame] = []
    dates = _source_dates(source)
    for position, day in enumerate(dates, start=1):
        date = day.strftime("%Y-%m-%d")
        key = unified_dataset_key(symbol, interval, date, prefix)
        frame = load_s3_parquet(s3, bucket, key)
        if frame is None:
            logger.warning("[%d/%d] missing s3://%s/%s", position, len(dates), bucket, key)
            continue
        frames.append(frame)
        if position == 1 or position % 50 == 0 or position == len(dates):
            logger.info(
                "[%d/%d] loaded date=%s rows=%d total_rows=%d",
                position,
                len(dates),
                date,
                len(frame),
                sum(len(item) for item in frames),
            )

    if not frames:
        raise FileNotFoundError("No dataset partitions were loaded")
    return pd.concat(frames, ignore_index=True)


def detect_features(
    dataset: pd.DataFrame,
    timestamp_columns: list[str],
    target_columns: dict[str, str],
) -> list[str]:
    excluded = set(timestamp_columns) | set(target_columns.values())
    features = []
    for column in dataset.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(dataset[column]) or pd.api.types.is_bool_dtype(
            dataset[column]
        ):
            features.append(column)
    if not features:
        raise ValueError("No numeric feature columns were detected")
    return features


def finite_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return frame["x"].to_numpy(dtype="float64"), frame["y"].to_numpy(dtype="float64")


def safe_skew(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").skew())


def safe_kurtosis(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").kurt())


def build_feature_summary(dataset: pd.DataFrame, features: list[str], logger: logging.Logger) -> pd.DataFrame:
    rows = []
    for feature in features:
        try:
            series = pd.to_numeric(dataset[feature], errors="coerce")
            rows.append(
                {
                    "feature": feature,
                    "observations": int(series.notna().sum()),
                    "missing": int(series.isna().sum()),
                    "unique_values": int(series.nunique(dropna=True)),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1)),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                    "skew": safe_skew(series),
                    "kurtosis": safe_kurtosis(series),
                    "error": "",
                }
            )
        except Exception as exc:
            logger.exception("feature_summary failed feature=%s", feature)
            rows.append({"feature": feature, "error": str(exc)})
    return pd.DataFrame(rows)


def _safe_stat_result(method, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan
    if stats is None:
        return float(pd.Series(x).corr(pd.Series(y), method=method)), np.nan
    if method == "pearson":
        result = stats.pearsonr(x, y)
    elif method == "spearman":
        result = stats.spearmanr(x, y)
    else:
        raise ValueError(method)
    return float(result.statistic), float(result.pvalue)


def _quantile_codes(values: np.ndarray, bins: int) -> np.ndarray:
    series = pd.Series(values)
    if series.nunique(dropna=True) <= 1:
        return np.zeros(len(series), dtype="int64")
    try:
        codes = pd.qcut(series, q=bins, labels=False, duplicates="drop")
    except ValueError:
        codes = pd.cut(series.rank(method="average"), bins=bins, labels=False)
    return pd.Series(codes).fillna(-1).astype("int64").to_numpy()


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    if len(x) < 3:
        return np.nan
    x_codes = _quantile_codes(x, bins)
    y_codes = _quantile_codes(y, bins)
    valid = (x_codes >= 0) & (y_codes >= 0)
    if valid.sum() < 3:
        return np.nan
    table = pd.crosstab(x_codes[valid], y_codes[valid]).to_numpy(dtype="float64")
    total = table.sum()
    if total <= 0:
        return np.nan
    probability = table / total
    px = probability.sum(axis=1, keepdims=True)
    py = probability.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = probability > 0
    return float((probability[mask] * np.log(probability[mask] / expected[mask])).sum())


def distance_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_rows: int | None,
    random_seed: int,
) -> tuple[float, int, bool]:
    if len(x) < 3:
        return np.nan, len(x), False
    sampled = False
    if max_rows and len(x) > max_rows:
        rng = np.random.default_rng(random_seed)
        index = rng.choice(len(x), size=max_rows, replace=False)
        x = x[index]
        y = y[index]
        sampled = True
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, len(x), sampled

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    a = np.abs(x - x.T)
    b = np.abs(y - y.T)
    a -= a.mean(axis=0, keepdims=True)
    a -= a.mean(axis=1, keepdims=True)
    a += a.mean()
    b -= b.mean(axis=0, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    b += b.mean()
    dcov2 = np.mean(a * b)
    dvar_x = np.mean(a * a)
    dvar_y = np.mean(b * b)
    if dvar_x <= 0 or dvar_y <= 0:
        return np.nan, len(x), sampled
    return float(math.sqrt(max(dcov2, 0.0) / math.sqrt(dvar_x * dvar_y))), len(x), sampled


def build_feature_metrics(
    dataset: pd.DataFrame,
    features: list[str],
    target_column: str,
    config: dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    analysis = config["analysis"]
    rows = []
    for feature in features:
        try:
            x, y = finite_pair(dataset[feature], dataset[target_column])
            pearson, pearson_p = _safe_stat_result("pearson", x, y)
            spearman, spearman_p = _safe_stat_result("spearman", x, y)
            mi = mutual_information(x, y, int(analysis.get("mutual_information_bins", 20)))
            dcor, dcor_rows, dcor_sampled = distance_correlation(
                x,
                y,
                analysis.get("distance_correlation_max_rows"),
                int(analysis.get("random_seed", 42)),
            )
            rows.append(
                {
                    "feature": feature,
                    "observations": int(len(x)),
                    "pearson": pearson,
                    "pearson_p_value": pearson_p,
                    "spearman": spearman,
                    "spearman_p_value": spearman_p,
                    "mutual_information": mi,
                    "distance_correlation": dcor,
                    "distance_correlation_rows": int(dcor_rows),
                    "distance_correlation_sampled": bool(dcor_sampled),
                    "error": "",
                }
            )
        except Exception as exc:
            logger.exception("feature_metrics failed target=%s feature=%s", target_column, feature)
            rows.append({"feature": feature, "error": str(exc)})
    return pd.DataFrame(rows)


def assign_feature_bins(feature: pd.Series, deciles: int) -> pd.DataFrame:
    values = pd.to_numeric(feature, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = values.dropna()
    result = pd.DataFrame(index=feature.index)
    result["bin"] = np.nan
    result["bin_left"] = np.nan
    result["bin_right"] = np.nan
    result["bin_label"] = pd.Series([None] * len(feature), index=feature.index, dtype="object")
    if valid.empty:
        return result
    if valid.nunique(dropna=True) <= 1:
        result.loc[valid.index, ["bin", "bin_left", "bin_right"]] = [
            0,
            float(valid.iloc[0]),
            float(valid.iloc[0]),
        ]
        result.loc[valid.index, "bin_label"] = str(valid.iloc[0])
        return result
    bins = pd.qcut(valid, q=deciles, duplicates="drop")
    categories = bins.cat.categories
    code_by_index = bins.cat.codes
    result.loc[valid.index, "bin"] = code_by_index.astype("int64").to_numpy()
    for code, interval in enumerate(categories):
        mask = valid.index[code_by_index == code]
        result.loc[mask, "bin_left"] = float(interval.left)
        result.loc[mask, "bin_right"] = float(interval.right)
        result.loc[mask, "bin_label"] = str(interval)
    return result


def _target_distribution_stats(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "observations": int(len(clean)),
        "target_mean": float(clean.mean()),
        "target_median": float(clean.median()),
        "target_std": float(clean.std(ddof=1)),
        "target_min": float(clean.min()),
        "target_max": float(clean.max()),
        "target_p05": float(clean.quantile(0.05)),
        "target_p25": float(clean.quantile(0.25)),
        "target_p75": float(clean.quantile(0.75)),
        "target_p95": float(clean.quantile(0.95)),
        "target_skew": safe_skew(clean),
        "target_kurtosis": safe_kurtosis(clean),
        "prob_positive_return": float((clean > 0).mean()),
        "prob_negative_return": float((clean < 0).mean()),
    }


def build_feature_bins(
    dataset: pd.DataFrame,
    features: list[str],
    target_column: str,
    config: dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    analysis = config["analysis"]
    deciles = int(analysis.get("deciles", 10))
    thresholds = [float(item) for item in analysis.get("return_thresholds", [])]
    rows = []
    target = pd.to_numeric(dataset[target_column], errors="coerce")
    for feature in features:
        try:
            bins = assign_feature_bins(dataset[feature], deciles)
            frame = pd.DataFrame({"target": target, **bins.to_dict("series")}).dropna(
                subset=["target", "bin"]
            )
            for bin_id, group in frame.groupby("bin", sort=True):
                clean_target = group["target"].dropna()
                row = {
                    "feature": feature,
                    "bin": int(bin_id),
                    "bin_left": float(group["bin_left"].iloc[0]),
                    "bin_right": float(group["bin_right"].iloc[0]),
                    "bin_label": str(group["bin_label"].iloc[0]),
                    "error": "",
                }
                row.update(_target_distribution_stats(clean_target))
                for threshold in thresholds:
                    suffix = str(threshold).replace(".", "_")
                    row[f"prob_abs_return_gt_{suffix}"] = float(
                        clean_target.abs().gt(threshold).mean()
                    )
                rows.append(row)
        except Exception as exc:
            logger.exception("feature_bins failed target=%s feature=%s", target_column, feature)
            rows.append({"feature": feature, "error": str(exc)})
    return pd.DataFrame(rows)


def build_feature_tests(
    dataset: pd.DataFrame,
    features: list[str],
    target_column: str,
    config: dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    deciles = int(config["analysis"].get("deciles", 10))
    rows = []
    target = pd.to_numeric(dataset[target_column], errors="coerce")
    for feature in features:
        try:
            bins = assign_feature_bins(dataset[feature], deciles)
            frame = pd.DataFrame({"target": target, "bin": bins["bin"]}).dropna()
            groups = [
                group["target"].to_numpy(dtype="float64")
                for _, group in frame.groupby("bin", sort=True)
                if len(group) >= 2
            ]
            row = {"feature": feature, "groups": int(len(groups)), "error": ""}
            if len(groups) < 2 or stats is None:
                row.update(
                    {
                        "ks_statistic": np.nan,
                        "ks_p_value": np.nan,
                        "anderson_darling_statistic": np.nan,
                        "anderson_darling_p_value": np.nan,
                    }
                )
            else:
                first = groups[0]
                last = groups[-1]
                ks = stats.ks_2samp(first, last)
                ad = stats.anderson_ksamp(groups)
                row.update(
                    {
                        "ks_statistic": float(ks.statistic),
                        "ks_p_value": float(ks.pvalue),
                        "anderson_darling_statistic": float(ad.statistic),
                        "anderson_darling_p_value": float(getattr(ad, "pvalue", np.nan)),
                    }
                )
            rows.append(row)
        except Exception as exc:
            logger.exception("feature_tests failed target=%s feature=%s", target_column, feature)
            rows.append({"feature": feature, "error": str(exc)})
    return pd.DataFrame(rows)


def build_correlation_pairs(dataset: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frame = dataset[features].apply(pd.to_numeric, errors="coerce")
    pearson = frame.corr(method="pearson")
    spearman = frame.corr(method="spearman")
    rows = []
    for left_index, feature_1 in enumerate(features):
        for feature_2 in features[left_index + 1 :]:
            rows.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "pearson": float(pearson.loc[feature_1, feature_2]),
                    "spearman": float(spearman.loc[feature_1, feature_2]),
                }
            )
    return pd.DataFrame(rows)


def write_parquet_local(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def upload_parquet(
    s3,
    bucket: str,
    key: str,
    frame: pd.DataFrame,
    dataset_kind: str,
) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, engine="pyarrow", compression="zstd")
    payload = buffer.getvalue()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType=PARQUET_CONTENT_TYPE,
        Metadata={
            "schema-version": SCHEMA_VERSION,
            "dataset-kind": dataset_kind,
            "rows": str(len(frame)),
        },
    )


def save_result(
    frame: pd.DataFrame,
    relative_path: str,
    config: dict[str, Any],
    logger: logging.Logger,
    s3=None,
) -> None:
    output = config["output"]
    local_dir = Path(output.get("local_dir", "analysis/feature_informativeness"))
    local_path = local_dir / relative_path
    write_parquet_local(frame, local_path)
    logger.info("wrote local path=%s rows=%d", local_path, len(frame))

    if output.get("upload_to_s3", True):
        if s3 is None:
            s3 = make_s3_client()
        normalized_relative_path = relative_path.replace("\\", "/")
        key = f"{output['prefix'].strip('/')}/{normalized_relative_path}"
        upload_parquet(s3, output["bucket"], key, frame, Path(relative_path).stem)
        logger.info("uploaded s3://%s/%s rows=%d", output["bucket"], key, len(frame))


def validate_targets(dataset: pd.DataFrame, targets: dict[str, str]) -> None:
    missing = [column for column in targets.values() if column not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing target columns: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze feature informativeness")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    config = load_config(args.config)
    if args.no_upload:
        config["output"]["upload_to_s3"] = False

    with timed_step(logger, "load_dataset"):
        dataset = load_dataset(config, logger)
    logger.info("dataset rows=%d columns=%d", len(dataset), len(dataset.columns))

    targets = config["analysis"]["targets"]
    validate_targets(dataset, targets)
    features = detect_features(
        dataset,
        config["analysis"].get("timestamp_columns", ["timestamp"]),
        targets,
    )
    logger.info("detected features=%d targets=%s", len(features), list(targets))
    s3 = make_s3_client() if config["output"].get("upload_to_s3", True) else None

    with timed_step(logger, "feature_summary"):
        summary = build_feature_summary(dataset, features, logger)
        save_result(summary, "feature_summary.parquet", config, logger, s3=s3)

    with timed_step(logger, "correlation_matrix"):
        correlation = build_correlation_pairs(dataset, features)
        save_result(correlation, "correlation_matrix.parquet", config, logger, s3=s3)

    for target_name, target_column in targets.items():
        logger.info("processing target=%s column=%s", target_name, target_column)
        target_prefix = target_name
        with timed_step(logger, f"{target_name}.feature_metrics"):
            metrics = build_feature_metrics(dataset, features, target_column, config, logger)
            save_result(
                metrics,
                f"{target_prefix}/feature_metrics.parquet",
                config,
                logger,
                s3=s3,
            )
        with timed_step(logger, f"{target_name}.feature_bins"):
            bins = build_feature_bins(dataset, features, target_column, config, logger)
            save_result(
                bins,
                f"{target_prefix}/feature_bins.parquet",
                config,
                logger,
                s3=s3,
            )
        with timed_step(logger, f"{target_name}.feature_tests"):
            tests = build_feature_tests(dataset, features, target_column, config, logger)
            save_result(
                tests,
                f"{target_prefix}/feature_tests.parquet",
                config,
                logger,
                s3=s3,
            )
        gc.collect()

    logger.info("completed")


if __name__ == "__main__":
    main()
