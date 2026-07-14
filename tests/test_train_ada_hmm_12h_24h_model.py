import unittest

import numpy as np
import pandas as pd

from prepare_ada_hmm_12h_24h_dataset import HMM_FEATURE_COLUMNS
from train_ada_hmm_12h_24h_model import (
    _build_state_profiles,
    _model_parameter_count,
    _prepare_training_frame,
)


class TrainAdaHmmModelTest(unittest.TestCase):
    def test_model_parameter_count_for_diag_covariance(self):
        self.assertEqual(
            _model_parameter_count(n_components=4, n_features=4, covariance_type="diag"),
            47,
        )

    def test_prepare_training_frame_sorts_and_drops_invalid_rows(self):
        timestamps = pd.to_datetime(
            ["2024-01-01T00:01:00Z", "2024-01-01T00:00:00Z"],
            utc=True,
        )
        dataset = pd.DataFrame(
            {
                "timestamp": timestamps,
                HMM_FEATURE_COLUMNS[0]: [1.0, np.nan],
                HMM_FEATURE_COLUMNS[1]: [2.0, 2.0],
                HMM_FEATURE_COLUMNS[2]: [3.0, 3.0],
                HMM_FEATURE_COLUMNS[3]: [4.0, 4.0],
            }
        )

        result = _prepare_training_frame(dataset)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["timestamp"], timestamps[0])

    def test_build_state_profiles_uses_original_feature_values(self):
        training_frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=4, freq="1min", tz="UTC"),
                HMM_FEATURE_COLUMNS[0]: [1.0, 3.0, 10.0, 14.0],
                HMM_FEATURE_COLUMNS[1]: [2.0, 4.0, 20.0, 24.0],
                HMM_FEATURE_COLUMNS[2]: [3.0, 5.0, 30.0, 34.0],
                HMM_FEATURE_COLUMNS[3]: [4.0, 6.0, 40.0, 44.0],
            }
        )
        states = np.array([0, 0, 1, 1])

        profiles = _build_state_profiles(training_frame, states, n_components=2)

        self.assertEqual(profiles.loc[0, "rows"], 2)
        self.assertEqual(profiles.loc[1, "rows"], 2)
        self.assertEqual(profiles.loc[0, f"{HMM_FEATURE_COLUMNS[0]}_mean"], 2.0)
        self.assertEqual(profiles.loc[1, f"{HMM_FEATURE_COLUMNS[0]}_mean"], 12.0)


if __name__ == "__main__":
    unittest.main()
