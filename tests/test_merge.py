"""Tests for merging response rows with parsed profile and portfolio data."""

import pandas as pd

from src.data.merge import merge_response_data


def test_merge_response_data_adds_profile_and_offer_columns() -> None:
    response_df = pd.DataFrame(
        [
            {
                "person": "p1",
                "offer_id": "offer-a",
                "offer_type": "bogo",
                "received_time": 0,
                "duration_days": 5,
                "label": 1,
            }
        ]
    )
    profile_df = pd.DataFrame(
        [
            {
                "person": "p1",
                "gender": "F",
                "age": 35.0,
                "income": 72000.0,
                "became_member_on": pd.Timestamp("2017-01-01"),
            }
        ]
    )
    portfolio_df = pd.DataFrame(
        [
            {
                "offer_id": "offer-a",
                "offer_type": "bogo",
                "duration": 5,
                "difficulty": 10,
                "reward": 5,
                "channel_web": 1,
                "channel_email": 1,
                "channel_mobile": 0,
                "channel_social": 0,
            }
        ]
    )

    merged = merge_response_data(response_df, profile_df, portfolio_df)

    assert len(merged) == 1
    assert merged.loc[0, "gender"] == "F"
    assert merged.loc[0, "difficulty"] == 10
    assert merged.loc[0, "reward"] == 5
    assert "duration" not in merged.columns
    assert "offer_type" in merged.columns


def test_merge_response_data_preserves_row_count_for_many_to_one_joins() -> None:
    response_df = pd.DataFrame(
        [
            {"person": "p1", "offer_id": "offer-a", "received_time": 0, "label": 1},
            {"person": "p1", "offer_id": "offer-b", "received_time": 10, "label": 0},
        ]
    )
    profile_df = pd.DataFrame([{"person": "p1", "gender": "M"}])
    portfolio_df = pd.DataFrame(
        [
            {"offer_id": "offer-a", "difficulty": 5},
            {"offer_id": "offer-b", "difficulty": 8},
        ]
    )

    merged = merge_response_data(response_df, profile_df, portfolio_df)

    assert len(merged) == 2
    assert merged["difficulty"].tolist() == [5, 8]
