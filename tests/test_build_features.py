"""Tests for final feature matrix assembly."""

import pandas as pd

from src.features.build_features import build_feature_matrix


def test_build_feature_matrix_joins_feature_tables_and_label() -> None:
    response_df = pd.DataFrame(
        [
            {"person": "p1", "offer_id": "offer-a", "received_time": 0, "label": 1},
        ]
    )
    demographic_df = pd.DataFrame(
        [
            {
                "person": "p1",
                "offer_id": "offer-a",
                "received_time": 0,
                "age_imputed": 40.0,
                "age_missing": 0,
                "income_imputed": 80000.0,
                "income_missing": 0,
                "membership_duration_days": 100,
                "gender_F": 0,
                "gender_M": 1,
                "gender_O": 0,
                "gender_unknown": 0,
            }
        ]
    )
    behavioral_df = pd.DataFrame(
        [
            {
                "person": "p1",
                "offer_id": "offer-a",
                "received_time": 0,
                "n_transactions_before": 3,
                "total_spend_before": 45.0,
                "avg_spend_before": 15.0,
                "days_since_last_transaction": 2.5,
                "offers_received_before": 1,
                "offers_completed_before": 1,
                "offer_completion_rate_before": 1.0,
            }
        ]
    )
    offer_df = pd.DataFrame(
        [
            {
                "person": "p1",
                "offer_id": "offer-a",
                "received_time": 0,
                "offer_type_bogo": 1,
                "offer_type_discount": 0,
                "difficulty": 10,
                "reward": 5,
                "duration_days": 7,
                "channel_web": 1,
                "channel_email": 1,
                "channel_mobile": 0,
                "channel_social": 0,
                "reward_to_difficulty_ratio": 0.5,
            }
        ]
    )

    features = build_feature_matrix(response_df, demographic_df, behavioral_df, offer_df)

    assert len(features) == 1
    assert features.loc[0, "label"] == 1
    assert features.loc[0, "income_to_difficulty_ratio"] == 8000.0
    assert features.loc[0, "reward_to_difficulty_ratio"] == 0.5


def test_build_feature_matrix_imputes_days_since_last_transaction() -> None:
    response_df = pd.DataFrame(
        [
            {"person": "p1", "offer_id": "offer-a", "received_time": 0, "label": 0},
            {"person": "p2", "offer_id": "offer-b", "received_time": 5, "label": 1},
        ]
    )
    demographic_df = pd.DataFrame(
        [
            {
                "person": "p1",
                "offer_id": "offer-a",
                "received_time": 0,
                "age_imputed": 30.0,
                "age_missing": 0,
                "income_imputed": 60000.0,
                "income_missing": 0,
                "membership_duration_days": 50,
                "gender_F": 1,
                "gender_M": 0,
                "gender_O": 0,
                "gender_unknown": 0,
            },
            {
                "person": "p2",
                "offer_id": "offer-b",
                "received_time": 5,
                "age_imputed": 35.0,
                "age_missing": 0,
                "income_imputed": 70000.0,
                "income_missing": 0,
                "membership_duration_days": 60,
                "gender_F": 0,
                "gender_M": 1,
                "gender_O": 0,
                "gender_unknown": 0,
            },
        ]
    )
    behavioral_df = pd.DataFrame(
        [
            {
                "person": "p1",
                "offer_id": "offer-a",
                "received_time": 0,
                "n_transactions_before": 0,
                "total_spend_before": 0.0,
                "avg_spend_before": 0.0,
                "days_since_last_transaction": pd.NA,
                "offers_received_before": 0,
                "offers_completed_before": 0,
                "offer_completion_rate_before": 0.0,
            },
            {
                "person": "p2",
                "offer_id": "offer-b",
                "received_time": 5,
                "n_transactions_before": 2,
                "total_spend_before": 20.0,
                "avg_spend_before": 10.0,
                "days_since_last_transaction": 4.0,
                "offers_received_before": 1,
                "offers_completed_before": 0,
                "offer_completion_rate_before": 0.0,
            },
        ]
    )
    offer_df = pd.DataFrame(
        [
            {
                "person": "p1",
                "offer_id": "offer-a",
                "received_time": 0,
                "offer_type_bogo": 1,
                "offer_type_discount": 0,
                "difficulty": 10,
                "reward": 5,
                "duration_days": 7,
                "channel_web": 1,
                "channel_email": 1,
                "channel_mobile": 0,
                "channel_social": 0,
                "reward_to_difficulty_ratio": 0.5,
            },
            {
                "person": "p2",
                "offer_id": "offer-b",
                "received_time": 5,
                "offer_type_bogo": 0,
                "offer_type_discount": 1,
                "difficulty": 20,
                "reward": 2,
                "duration_days": 10,
                "channel_web": 0,
                "channel_email": 1,
                "channel_mobile": 1,
                "channel_social": 0,
                "reward_to_difficulty_ratio": 0.1,
            },
        ]
    )

    features = build_feature_matrix(response_df, demographic_df, behavioral_df, offer_df)

    assert features["days_since_last_transaction"].isna().sum() == 0
    assert features.loc[0, "days_since_last_transaction_missing"] == 1
    assert features.loc[1, "days_since_last_transaction_missing"] == 0
    assert features.loc[0, "days_since_last_transaction"] == 5.0
