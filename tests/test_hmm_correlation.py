import unittest

import numpy as np
import pandas as pd

from build_hmm_correlation import OnlineCovariance


class OnlineCovarianceTest(unittest.TestCase):
    def test_matches_pandas_across_daily_batches(self):
        rng = np.random.default_rng(42)
        source = pd.DataFrame(rng.normal(size=(1000, 3)), columns=["a", "b", "c"])
        source["c"] = source["a"] * 0.8 + source["c"] * 0.2
        accumulator = OnlineCovariance(("a", "b", "c"))

        accumulator.update(source.iloc[:400])
        accumulator.update(source.iloc[400:])

        np.testing.assert_allclose(
            accumulator.correlation().to_numpy(),
            source.corr().to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(accumulator.count, len(source))


if __name__ == "__main__":
    unittest.main()
