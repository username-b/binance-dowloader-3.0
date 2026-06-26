"""Build a Pearson correlation matrix from daily HMM parquet partitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_hmm_datasets import HMM_FEATURE_COLUMNS, HMM_OUTPUT_PREFIX, hmm_dataset_key
from build_price_feature_day import load_s3_parquet, make_s3_client


# Inclusive UTC analysis range. For feature selection, use the train period only.
RUN_START_DATE = "2024-10-25"
RUN_END_DATE = "2026-02-01"

S3_BUCKET = "binance-data-downloader"
SYMBOL = "ADAUSDT"
OUTPUT_DIR = Path("analysis/hmm_correlation")
REPORT_EVERY_DAYS = 50

# Local files are always written. Enable this only if the results should also
# be stored in S3 alongside the HMM dataset.
UPLOAD_RESULTS_TO_S3 = False
S3_RESULTS_PREFIX = f"{HMM_OUTPUT_PREFIX}/statistics/correlation"


@dataclass
class OnlineCovariance:
    """Numerically stable batch-wise covariance accumulator."""

    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        size = len(self.columns)
        self.count = 0
        self.mean = np.zeros(size, dtype="float64")
        self.m2 = np.zeros((size, size), dtype="float64")

    def update(self, frame: pd.DataFrame) -> None:
        values = frame.loc[:, self.columns].to_numpy(dtype="float64", copy=False)
        if not np.isfinite(values).all():
            raise ValueError("HMM partition contains non-finite factor values")
        batch_count = len(values)
        if batch_count == 0:
            return

        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_m2 = centered.T @ centered
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + np.outer(delta, delta) * (
            self.count * batch_count / total
        )
        self.mean += delta * (batch_count / total)
        self.count = total

    def correlation(self) -> pd.DataFrame:
        if self.count < 2:
            raise ValueError("At least two HMM rows are required")
        covariance = self.m2 / (self.count - 1)
        standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        denominator = np.outer(standard_deviation, standard_deviation)
        correlation = np.full_like(covariance, np.nan)
        np.divide(covariance, denominator, out=correlation, where=denominator > 0)
        correlation = np.clip(correlation, -1.0, 1.0)
        return pd.DataFrame(correlation, index=self.columns, columns=self.columns)


def _validate_partition(frame: pd.DataFrame, date: str) -> pd.DataFrame:
    if frame is None:
        raise FileNotFoundError(f"HMM partition is missing for {date}")
    missing = set(HMM_FEATURE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"HMM partition {date} is missing factors: {sorted(missing)}")
    return frame


def _top_pairs(correlation: pd.DataFrame) -> pd.DataFrame:
    upper = np.triu(np.ones(correlation.shape, dtype=bool), k=1)
    pairs = correlation.where(upper).stack().rename("correlation").reset_index()
    pairs.columns = ["feature_1", "feature_2", "correlation"]
    pairs["absolute_correlation"] = pairs["correlation"].abs()
    return pairs.sort_values("absolute_correlation", ascending=False).reset_index(drop=True)


def _save_heatmap(correlation: pd.DataFrame, path: Path, rows: int) -> None:
    figure, axis = plt.subplots(figsize=(24, 22), constrained_layout=True)
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    positions = np.arange(len(correlation.columns))
    axis.set_xticks(positions, correlation.columns, rotation=90, fontsize=6)
    axis.set_yticks(positions, correlation.index, fontsize=6)
    axis.set_title(f"HMM feature Pearson correlations (rows={rows:,})")
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="Pearson r")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _upload_result(s3, local_path: Path, key: str, content_type: str) -> None:
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=local_path.read_bytes(),
        ContentType=content_type,
    )


def main() -> None:
    dates = pd.date_range(RUN_START_DATE, RUN_END_DATE, freq="D")
    s3 = make_s3_client()
    accumulator = OnlineCovariance(HMM_FEATURE_COLUMNS)

    print(f"Range: {RUN_START_DATE} -> {RUN_END_DATE} ({len(dates)} days)")
    for position, day in enumerate(dates, start=1):
        date = day.strftime("%Y-%m-%d")
        key = hmm_dataset_key(HMM_OUTPUT_PREFIX, SYMBOL, date)
        frame = _validate_partition(load_s3_parquet(s3, S3_BUCKET, key), date)
        accumulator.update(frame)
        if position == 1 or position % REPORT_EVERY_DAYS == 0 or position == len(dates):
            print(
                f"[{position}/{len(dates)}] {date} "
                f"rows={len(frame)} total={accumulator.count}"
            )

    correlation = accumulator.correlation()
    top_pairs = _top_pairs(correlation)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"hmm_correlation_{RUN_START_DATE}_{RUN_END_DATE}"
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    parquet_path = OUTPUT_DIR / f"{stem}.parquet"
    png_path = OUTPUT_DIR / f"{stem}.png"
    pairs_path = OUTPUT_DIR / f"{stem}_top_pairs.csv"
    correlation.to_csv(csv_path, index=True)
    correlation.to_parquet(parquet_path, index=True, compression="zstd")
    top_pairs.to_csv(pairs_path, index=False)
    _save_heatmap(correlation, png_path, accumulator.count)

    if UPLOAD_RESULTS_TO_S3:
        prefix = f"{S3_RESULTS_PREFIX}/start={RUN_START_DATE}/end={RUN_END_DATE}"
        _upload_result(s3, csv_path, f"{prefix}/correlation.csv", "text/csv")
        _upload_result(s3, parquet_path, f"{prefix}/correlation.parquet", "application/vnd.apache.parquet")
        _upload_result(s3, png_path, f"{prefix}/correlation.png", "image/png")
        _upload_result(s3, pairs_path, f"{prefix}/top_pairs.csv", "text/csv")

    zero_variance = correlation.columns[correlation.isna().all()].tolist()
    print(f"Completed: rows={accumulator.count}, factors={len(HMM_FEATURE_COLUMNS)}")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    if zero_variance:
        print(f"Zero-variance factors: {zero_variance}")


if __name__ == "__main__":
    main()
