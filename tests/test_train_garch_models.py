import math

import numpy as np
import pandas as pd

from analysis.garch_experiments.train_garch_models import (
    MODEL_SPECS,
    clean_returns,
    standardized_student_logpdf,
)


def test_model_specs_match_required_garch_family_grid():
    assert [(spec.name, spec.vol, spec.p, spec.o, spec.q) for spec in MODEL_SPECS] == [
        ("garch_1_1", "GARCH", 1, 0, 1),
        ("egarch_1_1", "EGARCH", 1, 1, 1),
        ("gjr_garch_1_1", "GARCH", 1, 1, 1),
        ("aparch_1_1", "APARCH", 1, 1, 1),
    ]


def test_clean_returns_drops_non_finite_values_and_keeps_timestamps_aligned():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="min"),
            "target": [0.1, np.nan, np.inf, -0.2, "bad"],
        }
    )

    returns, timestamps = clean_returns(
        frame,
        target_column="target",
        timestamp_column="timestamp",
    )

    assert returns.tolist() == [0.1, -0.2]
    assert timestamps.tolist() == [frame.loc[0, "timestamp"], frame.loc[3, "timestamp"]]


def test_standardized_student_logpdf_is_symmetric_and_finite_for_valid_nu():
    values = np.array([-1.0, 0.0, 1.0])

    logpdf = standardized_student_logpdf(values, nu=5.0)

    assert np.isfinite(logpdf).all()
    assert math.isclose(logpdf[0], logpdf[2])
    assert logpdf[1] > logpdf[0]
