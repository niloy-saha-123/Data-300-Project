"""Tests for customer-level feature engineering."""

import pandas as pd

from src.features.customer_features import (
    build_behavioral_features,
    build_demographic_features,
)


def test_build_demographic_features_creates_model_ready_columns() -> None:
    response_merged_df = pd.DataFrame(
        [
            {
                "person": "p1",
                "offer_id": "offer-a",
                "received_time": 0,
                "gender": "F",
                "age": 35.0,
                "income": 70000.0,
                "became_member_on": pd.Timestamp("2017-01-01"),
            },
            {
                "person": "p2",
                "offer_id": "offer-b",
                "received_time": 10,
                "gender": None,
                "age": pd.NA,
                "income": pd.NA,
                "became_member_on": pd.Timestamp("2017-01-03"),
            },
            {
                "person": "p3",
                "offer_id": "offer-c",
                "received_time": 20,
                "gender": "M",
                "age": 45.0,
                "income": 90000.0,
                "became_member_on": pd.Timestamp("2017-01-05"),
            },
        ]
    )

    features = build_demographic_features(
        response_merged_df, reference_date=pd.Timestamp("2017-01-10")
    )

    assert len(features) == 3
    assert features.columns.tolist() == [
        "person",
        "offer_id",
        "received_time",
        "age_imputed",
        "age_missing",
        "income_imputed",
        "income_missing",
        "membership_duration_days",
        "gender_F",
        "gender_M",
        "gender_O",
        "gender_unknown",
    ]
    assert features.loc[0, "gender_F"] == 1
    assert features.loc[1, "gender_unknown"] == 1
    assert features.loc[2, "gender_M"] == 1
    assert features.loc[1, "age_missing"] == 1
    assert features.loc[1, "income_missing"] == 1
    assert features.loc[1, "membership_duration_days"] == 7
    assert features.loc[1, "age_imputed"] == 40.0
    assert features.loc[1, "income_imputed"] == 80000.0


def test_build_demographic_features_imputes_income_by_gender_when_possible() -> None:
    response_merged_df = pd.DataFrame(
        [
            {
                "person": "p1",
                "offer_id": "offer-a",
                "received_time": 0,
                "gender": "F",
                "age": 30.0,
                "income": 70000.0,
                "became_member_on": pd.Timestamp("2017-01-01"),
            },
            {
                "person": "p2",
                "offer_id": "offer-b",
                "received_time": 1,
                "gender": "F",
                "age": 32.0,
                "income": pd.NA,
                "became_member_on": pd.Timestamp("2017-01-02"),
            },
            {
                "person": "p3",
                "offer_id": "offer-c",
                "received_time": 2,
                "gender": "M",
                "age": 40.0,
                "income": 90000.0,
                "became_member_on": pd.Timestamp("2017-01-03"),
            },
        ]
    )

    features = build_demographic_features(
        response_merged_df, reference_date=pd.Timestamp("2017-01-10")
    )

    assert features.loc[1, "income_imputed"] == 70000.0


def test_build_behavioral_features_uses_only_transactions_before_received_time() -> None:
    transcript_df = pd.DataFrame(
        [
            {"person": "p1", "event": "transaction", "time": 2, "amount": 10.0},
            {"person": "p1", "event": "transaction", "time": 10, "amount": 20.0},
            {"person": "p1", "event": "transaction", "time": 25, "amount": 100.0},
            {"person": "p2", "event": "transaction", "time": 3, "amount": 5.0},
        ]
    )
    response_df = pd.DataFrame(
        [
            {"person": "p1", "offer_id": "offer-a", "received_time": 5, "label": 0},
            {"person": "p1", "offer_id": "offer-b", "received_time": 20, "label": 1},
            {"person": "p2", "offer_id": "offer-c", "received_time": 1, "label": 0},
        ]
    )

    features = build_behavioral_features(transcript_df, response_df)

    assert len(features) == 3
    assert features.loc[0, "n_transactions_before"] == 1
    assert features.loc[0, "total_spend_before"] == 10.0
    assert features.loc[0, "avg_spend_before"] == 10.0
    assert features.loc[1, "n_transactions_before"] == 2
    assert features.loc[1, "total_spend_before"] == 30.0
    assert features.loc[1, "avg_spend_before"] == 15.0
    assert round(features.loc[1, "days_since_last_transaction"], 4) == round(
        (20 - 10) / 24.0, 4
    )
    assert features.loc[2, "n_transactions_before"] == 0
    assert pd.isna(features.loc[2, "days_since_last_transaction"])


def test_build_behavioral_features_tracks_prior_offer_history_by_received_time() -> None:
    transcript_df = pd.DataFrame(
        [{"person": "p1", "event": "transaction", "time": 1, "amount": 5.0}]
    )
    response_df = pd.DataFrame(
        [
            {"person": "p1", "offer_id": "offer-a", "received_time": 0, "label": 1},
            {"person": "p1", "offer_id": "offer-b", "received_time": 0, "label": 0},
            {"person": "p1", "offer_id": "offer-c", "received_time": 10, "label": 0},
        ]
    )

    features = build_behavioral_features(transcript_df, response_df)

    assert features["offers_received_before"].tolist() == [0, 0, 2]
    assert features["offers_completed_before"].tolist() == [0, 0, 1]
    assert features["offer_completion_rate_before"].tolist() == [0.0, 0.0, 0.5]
