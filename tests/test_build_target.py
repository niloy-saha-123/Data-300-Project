"""Tests for offer response target construction."""

import pandas as pd

from src.data.build_target import build_response_table, get_offer_events


def make_portfolio() -> pd.DataFrame:
    """Create a compact portfolio fixture for target-construction tests."""
    return pd.DataFrame(
        [
            {"offer_id": "offer-a", "offer_type": "bogo", "duration": 1},
            {"offer_id": "offer-b", "offer_type": "discount", "duration": 2},
            {"offer_id": "offer-info", "offer_type": "informational", "duration": 3},
        ]
    )


def test_get_offer_events_filters_out_transactions() -> None:
    transcript = pd.DataFrame(
        [
            {"person": "p1", "event": "offer received", "time": 0, "offer_id": "offer-a"},
            {"person": "p1", "event": "transaction", "time": 1, "offer_id": pd.NA},
            {"person": "p1", "event": "offer viewed", "time": 2, "offer_id": "offer-a"},
        ]
    )

    offer_events = get_offer_events(transcript)

    assert offer_events["event"].tolist() == ["offer received", "offer viewed"]


def test_build_response_table_labels_view_then_complete_as_positive() -> None:
    offer_events = pd.DataFrame(
        [
            {"person": "p1", "event": "offer received", "time": 0, "offer_id": "offer-a"},
            {"person": "p1", "event": "offer viewed", "time": 4, "offer_id": "offer-a"},
            {"person": "p1", "event": "offer completed", "time": 10, "offer_id": "offer-a"},
        ]
    )

    response = build_response_table(offer_events, make_portfolio())

    assert len(response) == 1
    assert bool(response.loc[0, "viewed"]) is True
    assert bool(response.loc[0, "completed_after_view"]) is True
    assert response.loc[0, "label"] == 1


def test_build_response_table_labels_completed_without_view_as_negative() -> None:
    offer_events = pd.DataFrame(
        [
            {"person": "p1", "event": "offer received", "time": 0, "offer_id": "offer-a"},
            {"person": "p1", "event": "offer completed", "time": 10, "offer_id": "offer-a"},
        ]
    )

    response = build_response_table(offer_events, make_portfolio())

    assert len(response) == 1
    assert bool(response.loc[0, "viewed"]) is False
    assert bool(response.loc[0, "completed_after_view"]) is False
    assert response.loc[0, "label"] == 0


def test_build_response_table_labels_view_without_complete_as_negative() -> None:
    offer_events = pd.DataFrame(
        [
            {"person": "p1", "event": "offer received", "time": 0, "offer_id": "offer-a"},
            {"person": "p1", "event": "offer viewed", "time": 5, "offer_id": "offer-a"},
        ]
    )

    response = build_response_table(offer_events, make_portfolio())

    assert len(response) == 1
    assert bool(response.loc[0, "viewed"]) is True
    assert bool(response.loc[0, "completed_after_view"]) is False
    assert response.loc[0, "label"] == 0


def test_build_response_table_excludes_informational_offers() -> None:
    offer_events = pd.DataFrame(
        [
            {"person": "p1", "event": "offer received", "time": 0, "offer_id": "offer-info"},
            {"person": "p1", "event": "offer viewed", "time": 5, "offer_id": "offer-info"},
        ]
    )

    response = build_response_table(offer_events, make_portfolio())

    assert response.empty


def test_build_response_table_keeps_repeat_receipts_separate() -> None:
    offer_events = pd.DataFrame(
        [
            {"person": "p1", "event": "offer received", "time": 0, "offer_id": "offer-a"},
            {"person": "p1", "event": "offer viewed", "time": 4, "offer_id": "offer-a"},
            {"person": "p1", "event": "offer completed", "time": 6, "offer_id": "offer-a"},
            {"person": "p1", "event": "offer received", "time": 30, "offer_id": "offer-a"},
        ]
    )

    response = build_response_table(offer_events, make_portfolio())

    assert len(response) == 2
    assert response["label"].tolist() == [1, 0]
