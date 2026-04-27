"""Tests for uplift helpers."""

import pandas as pd

from src.models.uplift import build_uplift_curve, get_uplift_feature_columns


def test_get_uplift_feature_columns_drops_labels_and_ids() -> None:
    features_df = pd.DataFrame(
        [
            {
                "person": "p1",
                "offer_id": "o1",
                "received_time": 0,
                "label": 1,
                "treatment": 1,
                "offer_type": "bogo",
                "age_imputed": 30.0,
                "income_imputed": 70000.0,
            }
        ]
    )

    assert get_uplift_feature_columns(features_df) == ["age_imputed", "income_imputed"]


def test_build_uplift_curve_adds_gain_and_qini_columns() -> None:
    scored_df = pd.DataFrame(
        [
            {"treatment": 1, "label": 1, "uplift_score": 0.6},
            {"treatment": 0, "label": 0, "uplift_score": 0.4},
            {"treatment": 1, "label": 0, "uplift_score": 0.2},
            {"treatment": 0, "label": 1, "uplift_score": 0.1},
        ]
    )

    curve_df = build_uplift_curve(scored_df)

    assert {"population_fraction", "cumulative_gain", "qini"} <= set(curve_df.columns)
    assert curve_df["population_fraction"].iloc[-1] == 1.0
