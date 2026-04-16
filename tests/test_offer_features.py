"""Tests for offer-level feature engineering."""

import pandas as pd

from src.features.offer_features import build_offer_features


def test_build_offer_features_creates_expected_offer_columns() -> None:
    response_merged_df = pd.DataFrame(
        [
            {
                "person": "p1",
                "offer_id": "offer-a",
                "received_time": 0,
                "offer_type": "bogo",
                "difficulty": 10,
                "reward": 5,
                "duration_days": 7,
                "channel_web": 1,
                "channel_email": 1,
                "channel_mobile": 0,
                "channel_social": 0,
            },
            {
                "person": "p2",
                "offer_id": "offer-b",
                "received_time": 10,
                "offer_type": "discount",
                "difficulty": 20,
                "reward": 2,
                "duration_days": 10,
                "channel_web": 0,
                "channel_email": 1,
                "channel_mobile": 1,
                "channel_social": 0,
            },
        ]
    )

    features = build_offer_features(response_merged_df)

    assert len(features) == 2
    assert features.columns.tolist() == [
        "person",
        "offer_id",
        "received_time",
        "offer_type_bogo",
        "offer_type_discount",
        "difficulty",
        "reward",
        "duration_days",
        "channel_web",
        "channel_email",
        "channel_mobile",
        "channel_social",
        "reward_to_difficulty_ratio",
    ]
    assert features.loc[0, "offer_type_bogo"] == 1
    assert features.loc[0, "offer_type_discount"] == 0
    assert features.loc[1, "offer_type_bogo"] == 0
    assert features.loc[1, "offer_type_discount"] == 1
    assert features.loc[0, "reward_to_difficulty_ratio"] == 0.5
    assert features.loc[1, "reward_to_difficulty_ratio"] == 0.1


def test_build_offer_features_handles_zero_difficulty_safely() -> None:
    response_merged_df = pd.DataFrame(
        [
            {
                "person": "p1",
                "offer_id": "offer-a",
                "received_time": 0,
                "offer_type": "discount",
                "difficulty": 0,
                "reward": 0,
                "duration_days": 4,
                "channel_web": 1,
                "channel_email": 1,
                "channel_mobile": 1,
                "channel_social": 0,
            }
        ]
    )

    features = build_offer_features(response_merged_df)

    assert features.loc[0, "reward_to_difficulty_ratio"] == 0.0
