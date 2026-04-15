"""Tests for raw data parsing helpers."""

from pathlib import Path

import pandas as pd

from src.data.parse_raw import load_portfolio, load_profile, load_transcript


def test_load_portfolio_expands_channels_and_renames_id(tmp_path: Path) -> None:
    sample = tmp_path / "portfolio.json"
    sample.write_text(
        '{"reward": 10, "channels": ["email", "mobile"], "difficulty": 10, '
        '"duration": 7.0, "offer_type": "bogo", "id": "offer-1"}\n'
        '{"reward": 0, "channels": ["web"], "difficulty": 0, '
        '"duration": 3.0, "offer_type": "informational", "id": "offer-2"}\n',
        encoding="utf-8",
    )

    portfolio = load_portfolio(sample)

    assert list(portfolio["offer_id"]) == ["offer-1", "offer-2"]
    assert "channels" not in portfolio.columns
    assert portfolio.loc[0, "channel_email"] == 1
    assert portfolio.loc[0, "channel_mobile"] == 1
    assert portfolio.loc[0, "channel_web"] == 0
    assert portfolio.loc[1, "channel_web"] == 1
    assert portfolio.loc[1, "channel_social"] == 0
    assert portfolio["duration"].tolist() == [7, 3]


def test_load_profile_cleans_member_date_and_missing_age(tmp_path: Path) -> None:
    sample = tmp_path / "profile.json"
    sample.write_text(
        '{"gender": null, "age": 118, "id": "person-2", '
        '"became_member_on": "20170212", "income": null}\n'
        '{"gender": "F", "age": 55, "id": "person-1", '
        '"became_member_on": "20170715", "income": 112000}\n',
        encoding="utf-8",
    )

    profile = load_profile(sample)

    assert list(profile["person"]) == ["person-1", "person-2"]
    assert pd.api.types.is_datetime64_any_dtype(profile["became_member_on"])
    assert pd.isna(profile.loc[1, "age"])
    assert pd.isna(profile.loc[1, "income"])
    assert pd.isna(profile.loc[1, "gender"])


def test_load_transcript_flattens_value_payload(tmp_path: Path) -> None:
    sample = tmp_path / "transcript.json"
    sample.write_text(
        '{"person": "person-1", "event": "offer received", '
        '"value": {"offer id": "offer-1"}, "time": 0}\n'
        '{"person": "person-1", "event": "transaction", '
        '"value": {"amount": 12.5}, "time": 10}\n'
        '{"person": "person-1", "event": "offer completed", '
        '"value": {"offer_id": "offer-1", "reward": 5}, "time": 20}\n',
        encoding="utf-8",
    )

    transcript = load_transcript(sample)

    assert list(transcript.columns) == [
        "person",
        "event",
        "time",
        "offer_id",
        "amount",
        "reward",
    ]
    assert transcript.loc[0, "offer_id"] == "offer-1"
    assert pd.isna(transcript.loc[0, "amount"])
    assert transcript.loc[1, "amount"] == 12.5
    assert pd.isna(transcript.loc[1, "offer_id"])
    assert transcript.loc[2, "offer_id"] == "offer-1"
    assert transcript.loc[2, "reward"] == 5
