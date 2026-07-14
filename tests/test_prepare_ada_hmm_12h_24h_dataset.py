import unittest

import numpy as np
import pandas as pd

from prepare_ada_hmm_12h_24h_dataset import (
    HMM_FEATURE_COLUMNS,
    TWELVE_HOURS,
    TWENTY_FOUR_HOURS,
    build_ada_hmm_12h_24h_dataset,
)


def make_klines(timestamps, close):
    close = np.asarray(close, dtype="float64")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        }
    )


class PrepareAdaHmmDatasetTest(unittest.TestCase):
    def test_selected_feature_formulas(self):
        timestamps = pd.date_range(
            "2024-01-01",
            periods=TWENTY_FOUR_HOURS + 2,
            freq="1min",
            tz="UTC",
        )
        minute_returns = np.full(len(timestamps) - 1, 0.001, dtype="float64")
        close = np.exp(np.r_[0.0, minute_returns.cumsum()])

        result = build_ada_hmm_12h_24h_dataset(make_klines(timestamps, close))

        self.assertEqual(result.columns.tolist(), ["timestamp", *HMM_FEATURE_COLUMNS])
        self.assertEqual(len(result), 2)
        row = result.iloc[0]
        source_position = TWENTY_FOUR_HOURS
        self.assertEqual(row["timestamp"], timestamps[source_position])
        self.assertAlmostEqual(row["ada_cum_return_12h"], TWELVE_HOURS * 0.001)
        self.assertAlmostEqual(
            row["ada_realized_volatility_12h"],
            np.sqrt(TWELVE_HOURS) * 0.001,
        )

        expected_efficiency = (
            close[source_position] - close[source_position - TWENTY_FOUR_HOURS]
        ) / np.diff(close[source_position - TWENTY_FOUR_HOURS : source_position + 1]).sum()
        self.assertAlmostEqual(row["ada_kaufman_efficiency_24h"], expected_efficiency)

        window_close = close[source_position]
        window_low = (close[source_position - TWELVE_HOURS + 1 : source_position + 1] - 0.5).min()
        window_high = (close[source_position - TWELVE_HOURS + 1 : source_position + 1] + 0.5).max()
        expected_close_position = (window_close - window_low) / (window_high - window_low)
        self.assertAlmostEqual(row["ada_close_position_12h"], expected_close_position)

    def test_reindexes_missing_minutes_before_rolling(self):
        timestamps = pd.to_datetime(
            ["2024-01-01T00:00:00Z", "2024-01-01T00:02:00Z"],
            utc=True,
        )
        result = build_ada_hmm_12h_24h_dataset(
            make_klines(timestamps, [100.0, 101.0]),
            dropna=False,
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(result["timestamp"].tolist(), pd.date_range(
            "2024-01-01",
            periods=3,
            freq="1min",
            tz="UTC",
        ).tolist())

    def test_flat_windows_have_zero_kaufman_and_middle_close_position(self):
        timestamps = pd.date_range(
            "2024-01-01",
            periods=TWENTY_FOUR_HOURS + 2,
            freq="1min",
            tz="UTC",
        )
        result = build_ada_hmm_12h_24h_dataset(make_klines(timestamps, np.ones(len(timestamps))))

        row = result.iloc[0]
        self.assertEqual(row["ada_kaufman_efficiency_24h"], 0.0)
        self.assertEqual(row["ada_close_position_12h"], 0.5)


if __name__ == "__main__":
    unittest.main()
