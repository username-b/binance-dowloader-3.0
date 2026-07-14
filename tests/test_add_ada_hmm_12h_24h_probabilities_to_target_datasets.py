import unittest

import numpy as np
import pandas as pd

from add_ada_hmm_12h_24h_probabilities_to_target_datasets import (
    _probability_columns,
    add_probabilities_to_target,
    build_probability_frame,
)
from prepare_ada_hmm_12h_24h_dataset import HMM_FEATURE_COLUMNS


class DummyScaler:
    def transform(self, values):
        return values


class DummyModel:
    n_components = 2

    def predict_proba(self, values):
        first = values[:, 0]
        probabilities = np.column_stack([first, 1.0 - first])
        return probabilities


class AddAdaHmmProbabilitiesTest(unittest.TestCase):
    def test_probability_columns(self):
        self.assertEqual(
            _probability_columns(2, "ada_hmm_12h_24h"),
            ["ada_hmm_12h_24h_prob_state_0", "ada_hmm_12h_24h_prob_state_1"],
        )

    def test_build_probability_frame(self):
        timestamps = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        hmm_dataset = pd.DataFrame(
            {
                "timestamp": timestamps,
                HMM_FEATURE_COLUMNS[0]: [0.25, 0.75],
                HMM_FEATURE_COLUMNS[1]: [1.0, 1.0],
                HMM_FEATURE_COLUMNS[2]: [1.0, 1.0],
                HMM_FEATURE_COLUMNS[3]: [1.0, 1.0],
            }
        )
        bundle = {
            "model": DummyModel(),
            "scaler": DummyScaler(),
            "features": list(HMM_FEATURE_COLUMNS),
        }

        probabilities, diagnostics = build_probability_frame(
            hmm_dataset,
            bundle,
            "ada_hmm_12h_24h",
        )

        self.assertEqual(len(probabilities), 2)
        self.assertEqual(diagnostics["valid_probability_rows"], 2)
        self.assertAlmostEqual(
            probabilities.iloc[0]["ada_hmm_12h_24h_prob_state_0"],
            0.25,
        )
        self.assertAlmostEqual(
            probabilities.iloc[1]["ada_hmm_12h_24h_prob_state_1"],
            0.25,
        )

    def test_add_probabilities_to_target_left_joins_by_timestamp(self):
        timestamps = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        target = pd.DataFrame(
            {
                "timestamp": timestamps,
                "feature": [1.0, 2.0, 3.0],
                "target_log_return_10m": [0.1, 0.2, 0.3],
            }
        )
        probability_columns = _probability_columns(2, "ada_hmm_12h_24h")
        probabilities = pd.DataFrame(
            {
                "timestamp": timestamps[:2],
                probability_columns[0]: [0.2, 0.8],
                probability_columns[1]: [0.8, 0.2],
            }
        )

        merged, diagnostics = add_probabilities_to_target(
            target,
            probabilities,
            probability_columns,
        )

        self.assertEqual(len(merged), 3)
        self.assertEqual(diagnostics["rows_with_new_hmm_probabilities"], 2)
        self.assertTrue(pd.isna(merged.iloc[2][probability_columns[0]]))
        self.assertEqual(merged.columns[-2:].tolist(), probability_columns)


if __name__ == "__main__":
    unittest.main()
