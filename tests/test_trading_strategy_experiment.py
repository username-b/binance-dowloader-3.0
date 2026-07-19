import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "analysis" / "trading_strategy_experiment" / "run_trading_strategy_experiment.py"
SPEC = importlib.util.spec_from_file_location("trading_strategy_experiment", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TradingStrategyExperimentTest(unittest.TestCase):
    def test_make_signal_prefers_stronger_direction(self):
        signal = MODULE.make_signal(
            np.array([0.7, 0.2, 0.7]),
            np.array([0.1, 0.8, 0.7]),
            0.6,
        )
        self.assertEqual(signal.tolist(), [1, -1, 1])

    def test_non_overlapping_charges_round_trip_fee(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=5, freq="min", tz="UTC"),
                "y_true": [0.002] * 5,
                "segment": ["evaluation"] * 5,
            }
        )
        trades, _, metrics = MODULE.simulate_non_overlapping(
            frame,
            np.ones(5, dtype=np.int8),
            horizon=2,
            fee_per_action=0.0005,
            strategy="test",
            segment="evaluation",
        )
        self.assertEqual(len(trades), 3)
        self.assertTrue(np.allclose(trades["commission"], 0.001))
        self.assertEqual(metrics["trades"], 3)

    def test_quantile_cdf_is_interpolated(self):
        frame = pd.DataFrame({"q05": [-2.0], "q25": [-1.0], "q50": [0.0], "q75": [1.0], "q95": [2.0]})
        self.assertTrue(np.allclose(MODULE.quantile_cdf_at(frame, 0.0), [0.5]))


if __name__ == "__main__":
    unittest.main()
