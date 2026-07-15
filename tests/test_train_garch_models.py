import math

import numpy as np
import pandas as pd

from analysis.garch_experiments.train_garch_models import (
    MODEL_SPECS,
    clean_returns,
    standardized_student_logpdf,
)
from analysis.garch_experiments.evaluate_point_garch_combinations import (
    build_point_garch_forecasts,
    evaluate_point_garch_combinations,
    select_garch_models,
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


def test_evaluate_point_garch_combinations_scores_hybrid_distribution():
    timestamps = pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC")
    point_predictions = {
        "point_a": pd.DataFrame(
            {
                "timestamp": timestamps,
                "y_true": [0.01, -0.02, 0.015, 0.0, 0.005],
                "y_pred": [0.008, -0.018, 0.010, 0.001, 0.004],
            }
        )
    }
    garch_results = pd.DataFrame({"model_name": ["garch_1_1"], "test_nll": [1.0], "nu": [6.0]})
    garch_predictions = pd.DataFrame(
        {
            "timestamp": timestamps,
            "model_name": "garch_1_1",
            "volatility_forecast": [0.01, 0.012, 0.011, 0.009, 0.010],
            "variance_forecast": [0.0001, 0.000144, 0.000121, 0.000081, 0.0001],
        }
    )

    results = evaluate_point_garch_combinations(
        point_predictions=point_predictions,
        garch_results=garch_results,
        garch_predictions=garch_predictions,
    )

    assert results.loc[0, "model_name"] == "point_a+garch_1_1_studentst"
    assert np.isfinite(results.loc[0, "NLL"])
    assert 0.0 <= results.loc[0, "Coverage90"] <= 1.0


def test_evaluate_point_garch_combinations_scores_all_garch_models_and_exports_forecasts():
    timestamps = pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC")
    point_predictions = {
        "point_a": pd.DataFrame(
            {
                "timestamp": timestamps,
                "y_true": [0.01, -0.02, 0.015, 0.0],
                "y_pred": [0.008, -0.018, 0.010, 0.001],
            }
        )
    }
    garch_results = pd.DataFrame(
        {
            "model_name": ["egarch_1_1", "garch_1_1"],
            "test_nll": [1.5, 1.0],
            "nu": [7.0, 6.0],
        }
    )
    garch_predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "model_name": model_name,
                    "volatility_forecast": sigma,
                    "variance_forecast": np.square(sigma),
                }
            )
            for model_name, sigma in [
                ("egarch_1_1", np.array([0.011, 0.013, 0.012, 0.010])),
                ("garch_1_1", np.array([0.010, 0.012, 0.011, 0.009])),
            ]
        ],
        ignore_index=True,
    )

    selected = select_garch_models(garch_results, selection="all")
    results = evaluate_point_garch_combinations(
        point_predictions=point_predictions,
        garch_results=garch_results,
        garch_predictions=garch_predictions,
    )
    forecasts = build_point_garch_forecasts(
        point_predictions=point_predictions,
        garch_results=garch_results,
        garch_predictions=garch_predictions,
    )

    assert selected == ["garch_1_1", "egarch_1_1"]
    assert set(results["garch_model"]) == {"garch_1_1", "egarch_1_1"}
    assert len(forecasts) == len(timestamps) * 2
    assert {"q05", "q25", "q50", "q75", "q95"}.issubset(forecasts.columns)
