"""Tests for budget simulation helpers."""

import pandas as pd

from src.models.budget_simulation import prepare_budget_frame, run_budget_simulation


def make_scores() -> pd.DataFrame:
    """Create a compact uplift score fixture."""
    return pd.DataFrame(
        [
            {"treatment": 1, "uplift_score": 0.30, "difficulty": 10.0, "reward": 2.0},
            {"treatment": 1, "uplift_score": 0.10, "difficulty": 10.0, "reward": 4.0},
            {"treatment": 1, "uplift_score": -0.05, "difficulty": 10.0, "reward": 2.0},
            {"treatment": 0, "uplift_score": 0.20, "difficulty": 0.0, "reward": 0.0},
        ]
    )


def test_prepare_budget_frame_filters_to_actionable_rows() -> None:
    budget_df = prepare_budget_frame(make_scores())

    assert len(budget_df) == 3
    assert budget_df["expected_incremental_response"].ge(0).all()


def test_run_budget_simulation_returns_summary_and_curve() -> None:
    payload = run_budget_simulation(make_scores())

    assert {"summary", "curve"} <= set(payload)
    assert payload["summary"]["candidate_rows"] == 3
    assert not payload["curve"].empty
