"""Tests for training helpers."""

import pandas as pd

from src.models.train import (
    choose_split_strategy,
    get_model_feature_columns,
    prepare_xy,
    split_feature_matrix,
)


def make_features_df() -> pd.DataFrame:
    """Create a compact feature matrix fixture."""
    rows = []
    for index in range(20):
        rows.append(
            {
                "person": f"p{index}",
                "offer_id": f"offer-{index % 2}",
                "received_time": float(index * 12),
                "label": index % 2,
                "age_imputed": 30 + index,
                "income_imputed": 50000 + (index * 1000),
                "difficulty": 10,
                "reward": 5,
            }
        )
    return pd.DataFrame(rows)


def test_choose_split_strategy_falls_back_to_random_without_late_holdout() -> None:
    features_df = make_features_df()

    assert choose_split_strategy(features_df) == "random"


def test_split_feature_matrix_random_strategy_preserves_all_rows() -> None:
    features_df = make_features_df()

    train_df, validation_df, test_df = split_feature_matrix(
        features_df, strategy="random", random_state=42
    )

    assert len(train_df) + len(validation_df) + len(test_df) == len(features_df)
    assert len(train_df) == 12
    assert len(validation_df) == 4
    assert len(test_df) == 4


def test_prepare_xy_drops_ids_and_target() -> None:
    features_df = make_features_df()

    X, y = prepare_xy(features_df)

    assert get_model_feature_columns(features_df) == [
        "age_imputed",
        "income_imputed",
        "difficulty",
        "reward",
    ]
    assert X.columns.tolist() == ["age_imputed", "income_imputed", "difficulty", "reward"]
    assert y.name == "label"
