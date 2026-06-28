import logging
import unittest

import numpy as np
import pandas as pd

from build_feature_informativeness import (
    assign_feature_bins,
    build_correlation_pairs,
    build_feature_bins,
    build_feature_metrics,
    build_feature_summary,
    detect_features,
    distance_correlation,
    mutual_information,
)


CONFIG = {
    "analysis": {
        "targets": {
            "target_10": "target_log_return_10m",
            "target_20": "target_log_return_20m",
            "target_30": "target_log_return_30m",
        },
        "timestamp_columns": ["timestamp"],
        "deciles": 4,
        "return_thresholds": [0.0025, 0.005],
        "mutual_information_bins": 4,
        "distance_correlation_max_rows": 100,
        "random_seed": 7,
    }
}


class FeatureInformativenessTest(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test_feature_informativeness")
        values = np.arange(1, 13, dtype="float64")
        self.dataset = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=len(values), freq="1min"),
                "feature_a": values,
                "feature_b": values[::-1],
                "feature_constant": 1.0,
                "target_log_return_10m": values * 0.001,
                "target_log_return_20m": values[::-1] * 0.001,
                "target_log_return_30m": np.sin(values) * 0.001,
            }
        )

    def test_detect_features_excludes_timestamp_and_targets(self):
        features = detect_features(
            self.dataset,
            CONFIG["analysis"]["timestamp_columns"],
            CONFIG["analysis"]["targets"],
        )
        self.assertEqual(features, ["feature_a", "feature_b", "feature_constant"])

    def test_feature_summary_contains_one_row_per_feature(self):
        summary = build_feature_summary(self.dataset, ["feature_a", "feature_b"], self.logger)
        self.assertEqual(summary["feature"].tolist(), ["feature_a", "feature_b"])
        self.assertEqual(summary.loc[0, "observations"], 12)
        self.assertAlmostEqual(summary.loc[0, "median"], 6.5)

    def test_metrics_capture_linear_relationship(self):
        metrics = build_feature_metrics(
            self.dataset,
            ["feature_a", "feature_b"],
            "target_log_return_10m",
            CONFIG,
            self.logger,
        ).set_index("feature")

        self.assertGreater(metrics.loc["feature_a", "pearson"], 0.99)
        self.assertLess(metrics.loc["feature_b", "pearson"], -0.99)
        self.assertGreater(metrics.loc["feature_a", "mutual_information"], 0.0)

    def test_bins_emit_distribution_rows_and_threshold_probabilities(self):
        bins = build_feature_bins(
            self.dataset,
            ["feature_a"],
            "target_log_return_10m",
            CONFIG,
            self.logger,
        )

        self.assertEqual(len(bins), 4)
        self.assertIn("prob_abs_return_gt_0_0025", bins.columns)
        self.assertEqual(bins["observations"].sum(), len(self.dataset))

    def test_constant_feature_gets_single_bin(self):
        bins = assign_feature_bins(self.dataset["feature_constant"], 10)
        self.assertEqual(bins["bin"].dropna().unique().tolist(), [0.0])
        self.assertEqual(bins["bin_left"].dropna().unique().tolist(), [1.0])

    def test_correlation_pairs_are_upper_triangle_list(self):
        pairs = build_correlation_pairs(self.dataset, ["feature_a", "feature_b", "feature_constant"])
        self.assertEqual(len(pairs), 3)
        pair = pairs[(pairs["feature_1"] == "feature_a") & (pairs["feature_2"] == "feature_b")]
        self.assertAlmostEqual(pair.iloc[0]["pearson"], -1.0)

    def test_distance_correlation_samples_when_configured(self):
        x = np.arange(1000, dtype="float64")
        y = x.copy()
        dcor, rows, sampled = distance_correlation(x, y, max_rows=50, random_seed=1)
        self.assertTrue(sampled)
        self.assertEqual(rows, 50)
        self.assertGreater(dcor, 0.99)

    def test_mutual_information_positive_for_dependent_series(self):
        x = np.repeat(np.arange(4), 10)
        y = x.copy()
        self.assertGreater(mutual_information(x, y, bins=4), 0.0)


if __name__ == "__main__":
    unittest.main()
