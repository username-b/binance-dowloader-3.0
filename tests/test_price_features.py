import unittest

import numpy as np
import pandas as pd

from build_price_feature_day import unified_dataset_key, validate_day_dataset
from price_features import FEATURE_COLUMNS, TARGET_COLUMNS, build_price_features


def make_klines(timestamps, close, volume=None, trades=None):
    close = np.asarray(close, dtype="float64")
    if volume is None:
        volume = np.ones(len(close), dtype="float64")
    if trades is None:
        trades = np.ones(len(close), dtype="float64")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": volume,
            "trades": trades,
        }
    )


class PriceFeaturesTest(unittest.TestCase):
    def test_unified_dataset_s3_key(self):
        self.assertEqual(
            unified_dataset_key("ADAUSDT", "1m", "2020-02-01"),
            "features/unified_dataset/symbol=ADAUSDT/interval=1m/"
            "date=2020-02-01/data.parquet",
        )

    def test_range_validator_accepts_complete_interior_day(self):
        timestamps = pd.date_range("2020-02-02", periods=1440, freq="1min", tz="UTC")
        dataset = pd.DataFrame(
            {
                "timestamp": timestamps,
                **{column: 0.0 for column in FEATURE_COLUMNS},
                **{column: 0.0 for column in TARGET_COLUMNS},
            }
        )
        validate_day_dataset(dataset, "2020-02-02")

    def test_formulas_and_column_set(self):
        timestamps = pd.date_range("2024-01-01", periods=100, freq="1min", tz="UTC")
        close = np.exp(np.arange(100, dtype="float64") * 0.01)
        result = build_price_features(make_klines(timestamps, close))

        self.assertEqual(
            result.columns.tolist(), ["timestamp", *FEATURE_COLUMNS, *TARGET_COLUMNS]
        )
        row = result.iloc[90]
        for lag in (1, 2, 3, 5, 10, 15, 30, 60):
            self.assertAlmostEqual(row[f"ada_log_return_lag_{lag}"], 0.01)
        for window in (3, 5, 10, 20, 30, 60):
            self.assertAlmostEqual(row[f"ada_cum_return_{window}m"], window * 0.01)
        for window in (3, 5, 10, 20, 30):
            self.assertAlmostEqual(row[f"ada_return_acceleration_{window}m"], 0.0)
        for window in (5, 10, 20, 30, 60, 90):
            expected_efficiency = (close[90] - close[90 - window]) / np.diff(
                close[90 - window : 91]
            ).sum()
            self.assertAlmostEqual(
                row[f"ada_kaufman_efficiency_{window}m"], expected_efficiency
            )
            expected_window = close[90 - window + 1 : 91]
            expected_zscore = (
                close[90] - expected_window.mean()
            ) / expected_window.std(ddof=0)
            self.assertAlmostEqual(row[f"ada_close_zscore_{window}m"], expected_zscore)
        for window in (5, 10, 20, 30, 60):
            self.assertAlmostEqual(row[f"ada_return_std_{window}m"], 0.0)
            self.assertAlmostEqual(
                row[f"ada_realized_volatility_{window}m"],
                np.sqrt(window) * 0.01,
            )

    def test_acceleration_uses_return_k_minutes_ago(self):
        timestamps = pd.date_range("2024-01-01", periods=8, freq="1min", tz="UTC")
        minute_returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        close = np.exp(np.r_[0.0, minute_returns.cumsum()])
        result = build_price_features(make_klines(timestamps, close))

        self.assertAlmostEqual(
            result.iloc[7]["ada_return_acceleration_3m"],
            minute_returns[6] - minute_returns[3],
        )

    def test_flat_windows_have_zero_efficiency_and_zscore(self):
        timestamps = pd.date_range("2024-01-01", periods=100, freq="1min", tz="UTC")
        result = build_price_features(make_klines(timestamps, np.ones(100)))

        row = result.iloc[90]
        for window in (5, 10, 20, 30, 60, 90):
            self.assertEqual(row[f"ada_kaufman_efficiency_{window}m"], 0.0)
            self.assertEqual(row[f"ada_close_zscore_{window}m"], 0.0)
        for window in (5, 10, 20, 30, 60):
            self.assertEqual(row[f"ada_return_std_{window}m"], 0.0)
            self.assertEqual(row[f"ada_realized_volatility_{window}m"], 0.0)

    def test_volatility_uses_trailing_minute_returns(self):
        timestamps = pd.date_range("2024-01-01", periods=8, freq="1min", tz="UTC")
        minute_returns = np.array([0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07])
        close = np.exp(np.r_[0.0, minute_returns.cumsum()])
        result = build_price_features(make_klines(timestamps, close))

        trailing = minute_returns[-5:]
        row = result.iloc[-1]
        self.assertAlmostEqual(row["ada_return_std_5m"], trailing.std(ddof=1))
        self.assertAlmostEqual(
            row["ada_realized_volatility_5m"], np.sqrt(np.square(trailing).sum())
        )

    def test_missing_minute_does_not_create_return_across_gap(self):
        timestamps = pd.to_datetime(
            ["2024-01-01T00:00:00Z", "2024-01-01T00:02:00Z"], utc=True
        )
        result = build_price_features(
            make_klines(timestamps, [100.0, 110.0])
        ).set_index("timestamp")

        self.assertTrue(pd.isna(result.loc[timestamps[1], "ada_log_return_lag_1"]))

    def test_rejects_duplicate_timestamps(self):
        timestamp = pd.Timestamp("2024-01-01", tz="UTC")
        with self.assertRaisesRegex(ValueError, "duplicate timestamps"):
            build_price_features(
                make_klines([timestamp, timestamp], [1.0, 2.0])
            )

    def test_candle_geometry_and_high_close_ratio(self):
        timestamps = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        close_positions = np.array([0.9, 0.8, 0.7, 1.0, 0.0])
        low = np.full(5, 90.0)
        high = np.full(5, 110.0)
        close = low + close_positions * (high - low)
        open_price = np.full(5, 100.0)
        klines = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.ones(5),
                "trades": np.ones(5),
            }
        )

        row = build_price_features(klines).iloc[-1]
        candle_open = open_price[0]
        candle_high = high.max()
        candle_low = low.min()
        candle_close = close[-1]
        candle_range = candle_high - candle_low
        self.assertAlmostEqual(
            row["ada_upper_wick_5m"],
            (candle_high - max(candle_open, candle_close)) / candle_range,
        )
        self.assertAlmostEqual(
            row["ada_lower_wick_5m"],
            (min(candle_open, candle_close) - candle_low) / candle_range,
        )
        self.assertAlmostEqual(row["ada_log_range_5m"], np.log(110.0 / 90.0))
        self.assertAlmostEqual(
            row["ada_close_position_5m"],
            (candle_close - candle_low) / candle_range,
        )
        self.assertAlmostEqual(row["ada_high_close_ratio_5m"], 3.0 / 5.0)

    def test_volume_sum_and_zscore(self):
        timestamps = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
        volume = np.arange(1.0, 11.0)
        result = build_price_features(
            make_klines(timestamps, np.full(10, 100.0), volume=volume)
        )

        row = result.iloc[-1]
        self.assertAlmostEqual(row["ada_volume_sum_10m"], volume.sum())
        self.assertAlmostEqual(
            row["ada_volume_zscore_10m"],
            (volume[-1] - volume.mean()) / volume.std(ddof=0),
        )

    def test_constant_volume_has_zero_zscore(self):
        timestamps = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
        result = build_price_features(
            make_klines(timestamps, np.full(10, 100.0), volume=np.full(10, 7.0))
        )

        row = result.iloc[-1]
        self.assertEqual(row["ada_volume_sum_10m"], 70.0)
        self.assertEqual(row["ada_volume_zscore_10m"], 0.0)

    def test_trades_per_minute_is_averaged_from_minute_klines(self):
        timestamps = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        trades = np.array([2, 3, 5, 7, 11])
        result = build_price_features(
            make_klines(timestamps, np.full(5, 100.0), trades=trades)
        )

        self.assertEqual(result.iloc[-1]["ada_trades_per_minute_5m"], trades.mean())

    def test_trade_flow_statistics_use_raw_non_aggregated_trades(self):
        timestamps = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        quantities = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        buyer_maker = np.array([False, False, False, False, False, True])
        trades = pd.DataFrame(
            {
                "timestamp": timestamps[[0, 1, 2, 3, 4, 4]],
                "qty": quantities,
                "quote_qty": quantities * 10.0,
                "is_buyer_maker": buyer_maker,
            }
        )
        result = build_price_features(
            make_klines(timestamps, np.full(5, 100.0)),
            trades=trades,
        )

        row = result.iloc[-1]
        weights = quantities / quantities.sum()
        expected_entropy = -(weights * np.log(weights)).sum() / np.log(len(weights))
        expected_simpson = 1.0 / (len(weights) * np.square(weights).sum())
        self.assertAlmostEqual(row["ada_aggression_delta_norm_5m"], -85.0 / 115.0)
        self.assertAlmostEqual(
            row["ada_large_trade_aggression_delta_norm_5m"], -95.0 / 105.0
        )
        self.assertAlmostEqual(row["ada_volume_entropy_norm_5m"], expected_entropy)
        self.assertAlmostEqual(row["ada_inverse_simpson_norm_5m"], expected_simpson)
        self.assertAlmostEqual(
            row["ada_average_trade_size_quote_5m"], 1150.0 / 6.0
        )
        self.assertAlmostEqual(row["ada_pressure_concentration_5m"], 105.0 / 85.0)

    def test_btc_features_use_aligned_btc_closes(self):
        timestamps = pd.date_range("2024-01-01", periods=100, freq="1min", tz="UTC")
        ada = make_klines(timestamps, np.full(100, 1.0))
        btc_close = np.exp(np.arange(100, dtype="float64") * 0.02)
        btc = make_klines(timestamps, btc_close)
        result = build_price_features(ada, btc_klines=btc)

        row = result.iloc[90]
        for window in (5, 10, 20, 30, 60):
            self.assertAlmostEqual(row[f"btc_log_return_{window}m"], window * 0.02)
        for window in (10, 20, 30, 60):
            self.assertAlmostEqual(row[f"btc_return_std_{window}m"], 0.0)
        for window in (30, 60):
            self.assertAlmostEqual(row[f"btc_kaufman_efficiency_{window}m"], 1.0)

    def test_forward_log_return_targets(self):
        timestamps = pd.date_range("2024-01-01", periods=40, freq="1min", tz="UTC")
        close = np.exp(np.arange(40, dtype="float64") * 0.01)
        result = build_price_features(make_klines(timestamps, close))

        self.assertAlmostEqual(result.iloc[0]["target_log_return_10m"], 0.10)
        self.assertAlmostEqual(result.iloc[0]["target_log_return_20m"], 0.20)
        self.assertAlmostEqual(result.iloc[0]["target_log_return_30m"], 0.30)
        self.assertTrue(pd.isna(result.iloc[-1]["target_log_return_10m"]))

    def test_time_features_use_utc_and_mark_weekend(self):
        timestamps = pd.date_range("2024-01-05 23:59", periods=3, freq="1min", tz="UTC")
        result = build_price_features(make_klines(timestamps, np.ones(3)))

        friday = result.iloc[0]
        saturday = result.iloc[1]
        self.assertFalse(friday["is_weekend"])
        self.assertTrue(saturday["is_weekend"])
        self.assertAlmostEqual(saturday["time_of_day_sin"], 0.0)
        self.assertAlmostEqual(saturday["time_of_day_cos"], 1.0)
        self.assertAlmostEqual(saturday["day_of_week_sin"], np.sin(2 * np.pi * 5 / 7))
        self.assertAlmostEqual(saturday["day_of_week_cos"], np.cos(2 * np.pi * 5 / 7))
        self.assertFalse(saturday["is_asia_open"])
        self.assertFalse(saturday["is_cboe_europe_open"])
        self.assertFalse(saturday["is_europe_open"])
        self.assertFalse(saturday["is_us_open"])

    def test_exchange_group_sessions_match_reference_diagram(self):
        timestamps = pd.date_range(
            "2025-03-12 00:00", "2025-03-12 21:00", freq="1min", tz="UTC"
        )
        result = build_price_features(
            make_klines(timestamps, np.full(len(timestamps), 100.0))
        ).set_index("timestamp")

        self.assertTrue(result.loc["2025-03-12 00:00", "is_asia_open"])
        self.assertTrue(result.loc["2025-03-12 07:00", "is_asia_open"])
        self.assertFalse(result.loc["2025-03-12 08:00", "is_asia_open"])
        self.assertTrue(result.loc["2025-03-12 07:00", "is_europe_open"])
        self.assertTrue(result.loc["2025-03-12 16:29", "is_europe_open"])
        self.assertFalse(result.loc["2025-03-12 16:30", "is_europe_open"])
        self.assertTrue(result.loc["2025-03-12 13:30", "is_us_open"])
        self.assertFalse(result.loc["2025-03-12 20:00", "is_us_open"])
        self.assertTrue(result.loc["2025-03-12 07:00", "is_cboe_europe_open"])
        self.assertTrue(result.loc["2025-03-12 20:59", "is_cboe_europe_open"])
        self.assertFalse(result.loc["2025-03-12 21:00", "is_cboe_europe_open"])


if __name__ == "__main__":
    unittest.main()
