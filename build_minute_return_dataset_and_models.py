import argparse
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv


BUCKET = "binance-data-downloader"
SYMBOL = "ADAUSDT"
INTERVAL = "1m"

EXCLUDED_FEATURE_GROUPS = {
    "return_15m_forward",
    "return_20m_forward",
    "return_25m_forward",
}

TARGET_GROUP = "return_1m_forward"
RAW_SYNTHETIC_GROUP = "sinthetic_data"


@dataclass(frozen=True)
class Source:
    name: str
    prefix: str
    is_target: bool = False
    is_raw_synthetic: bool = False


def build_s3_client():
    load_dotenv(dotenv_path=".env")
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("YC_ENDPOINT"),
        region_name=os.getenv("YC_REGION"),
        aws_access_key_id=os.getenv("YC_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("YC_SECRET_ACCESS_KEY"),
    )


def list_feature_groups(s3) -> list[str]:
    groups = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix="features/", Delimiter="/"):
        for item in page.get("CommonPrefixes", []):
            group = item["Prefix"].strip("/").split("/")[-1]
            if group and group not in EXCLUDED_FEATURE_GROUPS:
                groups.add(group)
    return sorted(groups)


def list_dates(s3, prefix: str) -> set[str]:
    dates = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            match = re.search(r"date=([0-9]{4}-[0-9]{2}-[0-9]{2})", obj["Key"])
            if match:
                dates.add(match.group(1))
    return dates


def read_parquet(s3, key: str) -> pd.DataFrame | None:
    try:
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
            return None
        raise
    return pd.read_parquet(io.BytesIO(body))


def key_for_source(source: Source, date: str) -> str:
    if source.is_raw_synthetic:
        return f"raw/{RAW_SYNTHETIC_GROUP}/date={date}/data.parquet"
    return (
        f"features/{source.name}/symbol={SYMBOL}/interval={INTERVAL}/"
        f"date={date}/data.parquet"
    )


def normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif "open_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    else:
        raise ValueError("No timestamp/open_time column found")
    return df


def prepare_source_frame(source: Source, df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_time(df)

    if "symbol" in df.columns:
        df = df[df["symbol"] == SYMBOL].copy()

    if source.is_target:
        if "return_1m" not in df.columns:
            raise ValueError(f"{source.name} has no return_1m column")
        df["log_return_1m"] = np.log1p(df["return_1m"])
        return df[["timestamp", "log_return_1m", "return_1m"]]

    drop_cols = {
        "date",
        "symbol",
        "interval",
        "timestamp",
        "open_time",
        "close_time",
    }
    value_cols = [col for col in df.columns if col not in drop_cols]
    out = df[["timestamp"] + value_cols].copy()
    rename = {col: f"{source.name}__{col}" for col in value_cols}
    return out.rename(columns=rename)


def build_sources(s3) -> list[Source]:
    groups = list_feature_groups(s3)
    if TARGET_GROUP not in groups:
        raise RuntimeError(f"Target group {TARGET_GROUP!r} was not found in S3")

    sources = [
        Source(name=group, prefix=f"features/{group}/", is_target=(group == TARGET_GROUP))
        for group in groups
    ]
    sources.append(
        Source(
            name=RAW_SYNTHETIC_GROUP,
            prefix=f"raw/{RAW_SYNTHETIC_GROUP}/",
            is_raw_synthetic=True,
        )
    )
    return sources


def build_dataset(args) -> tuple[pd.DataFrame, dict]:
    s3 = build_s3_client()
    sources = build_sources(s3)
    target_source = next(source for source in sources if source.is_target)
    target_prefix = f"features/{TARGET_GROUP}/symbol={SYMBOL}/interval={INTERVAL}/"
    dates = sorted(list_dates(s3, target_prefix))

    if args.start_date:
        dates = [date for date in dates if date >= args.start_date]
    if args.end_date:
        dates = [date for date in dates if date <= args.end_date]
    if args.max_days:
        dates = dates[: args.max_days]

    report = {
        "bucket": BUCKET,
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "excluded_feature_groups": sorted(EXCLUDED_FEATURE_GROUPS),
        "sources": [source.name for source in sources],
        "dates": {"count": len(dates), "start": dates[0], "end": dates[-1]},
        "missing_files": {},
        "rows_by_date": {},
    }

    frames = []
    for idx, date in enumerate(dates, start=1):
        day = None
        missing = []

        for source in [target_source] + [source for source in sources if not source.is_target]:
            key = key_for_source(source, date)
            df = read_parquet(s3, key)
            if df is None:
                missing.append(source.name)
                continue

            prepared = prepare_source_frame(source, df)
            if day is None:
                day = prepared
            else:
                day = day.merge(prepared, on="timestamp", how="left", validate="one_to_one")

        if day is None:
            missing.append("all")
            report["missing_files"][date] = missing
            continue

        if missing:
            report["missing_files"][date] = missing

        day.insert(0, "date", date)
        frames.append(day)
        report["rows_by_date"][date] = len(day)

        if args.progress_every and idx % args.progress_every == 0:
            print(f"processed {idx}/{len(dates)} days")

    dataset = pd.concat(frames, ignore_index=True)
    feature_cols = [
        col
        for col in dataset.columns
        if col not in {"date", "timestamp", "log_return_1m", "return_1m"}
    ]

    missing_by_column = dataset[feature_cols + ["log_return_1m"]].isna().sum()
    missing_by_column = missing_by_column[missing_by_column > 0].sort_values(ascending=False)
    rows_with_missing_features = dataset[feature_cols].isna().any(axis=1)
    report["dataset"] = {
        "rows": int(len(dataset)),
        "columns": int(dataset.shape[1]),
        "feature_columns": len(feature_cols),
        "target": "log_return_1m",
        "missing_cells": int(dataset[feature_cols + ["log_return_1m"]].isna().sum().sum()),
        "rows_with_missing_features": int(rows_with_missing_features.sum()),
        "missing_by_column": {col: int(value) for col, value in missing_by_column.items()},
        "rows_with_missing_features_sample": [
            {
                "date": str(row["date"]),
                "timestamp": str(row["timestamp"]),
            }
            for _, row in dataset.loc[rows_with_missing_features, ["date", "timestamp"]]
            .head(20)
            .iterrows()
        ],
    }

    if args.missing_feature_policy == "error" and rows_with_missing_features.any():
        raise RuntimeError(
            "Dataset contains rows with missing feature values. "
            "Use --missing-feature-policy keep or drop if this is expected."
        )

    if args.missing_feature_policy == "drop":
        before_rows = len(dataset)
        dataset = dataset.loc[~rows_with_missing_features].reset_index(drop=True)
        report["dataset"]["rows_before_missing_feature_drop"] = int(before_rows)
        report["dataset"]["rows_after_missing_feature_drop"] = int(len(dataset))
        report["dataset"]["dropped_rows_with_missing_features"] = int(before_rows - len(dataset))

    feature_cols_after_missing = [
        col
        for col in dataset.columns
        if col not in {"date", "timestamp", "log_return_1m", "return_1m"}
    ]
    constant_feature_cols = [
        col
        for col in feature_cols_after_missing
        if dataset[col].nunique(dropna=False) <= 1
    ]
    report["dataset"]["constant_feature_columns"] = constant_feature_cols
    report["dataset"]["constant_feature_columns_count"] = len(constant_feature_cols)

    if args.constant_feature_policy == "error" and constant_feature_cols:
        raise RuntimeError(
            "Dataset contains constant feature columns: "
            f"{constant_feature_cols}. Use --constant-feature-policy keep or drop "
            "if this is expected."
        )

    if args.constant_feature_policy == "drop" and constant_feature_cols:
        dataset = dataset.drop(columns=constant_feature_cols)
        report["dataset"]["dropped_constant_feature_columns"] = constant_feature_cols
        report["dataset"]["columns_after_constant_feature_drop"] = int(dataset.shape[1])
        report["dataset"]["feature_columns_after_constant_feature_drop"] = int(
            len(feature_cols_after_missing) - len(constant_feature_cols)
        )

    return dataset, report


def train_models(dataset: pd.DataFrame, args) -> dict:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(args.n_jobs, 1)))

    try:
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Model training requires scikit-learn. Install it with "
            "`pip install scikit-learn` or `pip install -r requirements.txt`."
        ) from exc

    data = dataset.sort_values("timestamp").copy()
    data = data[np.isfinite(data["log_return_1m"])]

    feature_cols = [
        col
        for col in data.columns
        if col not in {"date", "timestamp", "log_return_1m", "return_1m"}
    ]
    X = data[feature_cols]
    y = data["log_return_1m"]

    split = int(len(data) * args.train_fraction)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {
        "dummy_mean": make_pipeline(SimpleImputer(strategy="median"), DummyRegressor()),
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=args.ridge_alpha),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=args.hgb_iterations,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=42,
            ),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=args.trees,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=args.n_jobs,
            ),
        ),
    }

    results = {
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_end": str(data["timestamp"].iloc[split - 1]),
            "test_start": str(data["timestamp"].iloc[split]),
        },
        "models": {},
    }

    for name, model in models.items():
        print(f"training {name}")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        results["models"][name] = {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(rmse),
            "r2": float(r2_score(y_test, pred)),
            "directional_accuracy": float((np.sign(pred) == np.sign(y_test)).mean()),
        }

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--max-days", type=int)
    parser.add_argument("--output-dir", default="model_artifacts")
    parser.add_argument("--dataset-name", default="adausdt_minute_log_return_dataset.parquet")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument(
        "--missing-feature-policy",
        choices=["keep", "drop", "error"],
        default="keep",
        help="How to handle rows where at least one feature is missing.",
    )
    parser.add_argument(
        "--constant-feature-policy",
        choices=["keep", "drop", "error"],
        default="keep",
        help="How to handle feature columns with one unique value.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--hgb-iterations", type=int, default=200)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, report = build_dataset(args)
    dataset_path = output_dir / args.dataset_name
    report_path = output_dir / "dataset_report.json"

    dataset.to_parquet(dataset_path, index=False, compression="zstd")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"dataset saved: {dataset_path}")
    print(f"report saved: {report_path}")

    if not args.skip_models:
        results = train_models(dataset, args)
        metrics_path = output_dir / "model_metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
