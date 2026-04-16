"""Tests for customer-level feature engineering."""

import pandas as pd

from src.features.customer_features import build_demographic_features


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
