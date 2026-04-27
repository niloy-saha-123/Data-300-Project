"""Simulate budget allocation with uplift-ranked offers."""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path.cwd()
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.train import RANDOM_STATE
from src.models.uplift import REPORTS_DIR, UPLIFT_SCORES_FILE
from src.utils import plotting as _plotting  # noqa: F401


FIGURES_DIR = REPORTS_DIR / "figures"
BUDGET_SUMMARY_FILE = REPORTS_DIR / "budget_simulation.json"
BUDGET_CURVE_FILE = REPORTS_DIR / "budget_simulation_curve.csv"
BUDGET_FIGURE_FILE = FIGURES_DIR / "budget_allocation_comparison.png"
RANDOM_RUNS = 200
BUDGET_SHARE = 0.25


def prepare_budget_frame(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Keep actionable offers and score them by uplift per reward dollar."""
    actionable = scored_df.loc[scored_df["treatment"] == 1].copy()
    actionable["uplift_score_clipped"] = actionable["uplift_score"].clip(lower=0.0)
    actionable["expected_incremental_response"] = actionable["uplift_score_clipped"]
    actionable["expected_incremental_revenue"] = actionable["uplift_score_clipped"] * actionable[
        "difficulty"
    ]
    actionable["value_per_reward"] = actionable["expected_incremental_response"] / actionable[
        "reward"
    ].replace(0, np.nan)
    actionable["value_per_reward"] = actionable["value_per_reward"].fillna(0.0)
    return actionable.sort_values(
        ["value_per_reward", "uplift_score"], ascending=[False, False]
    ).reset_index(drop=True)


def select_under_budget(candidate_df: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Take rows in current order until budget is exhausted."""
    selected = candidate_df.copy()
    selected["cumulative_cost"] = selected["reward"].cumsum()
    selected["cumulative_response"] = selected["expected_incremental_response"].cumsum()
    return selected.loc[selected["cumulative_cost"] <= budget].reset_index(drop=True)


def interpolate_curve(selected_df: pd.DataFrame, budget_grid: np.ndarray) -> np.ndarray:
    """Interpolate cumulative uplift over a common budget grid."""
    if selected_df.empty:
        return np.zeros_like(budget_grid)
    x = np.concatenate([[0.0], selected_df["cumulative_cost"].to_numpy()])
    y = np.concatenate([[0.0], selected_df["cumulative_response"].to_numpy()])
    return np.interp(budget_grid, x, y, left=0.0, right=y[-1])


def run_budget_simulation(scored_df: pd.DataFrame) -> dict[str, object]:
    """Compare greedy allocation against a random baseline."""
    candidates = prepare_budget_frame(scored_df)
    budget = float(candidates["reward"].sum() * BUDGET_SHARE)
    greedy_selected = select_under_budget(
        candidates.loc[candidates["expected_incremental_response"] > 0].copy(), budget
    )

    rng = np.random.default_rng(RANDOM_STATE)
    budget_grid = np.linspace(0.0, budget, 25)
    random_curves = []
    random_totals = []
    for _ in range(RANDOM_RUNS):
        shuffled = candidates.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        selected = select_under_budget(shuffled, budget)
        random_curves.append(interpolate_curve(selected, budget_grid))
        random_totals.append(float(selected["expected_incremental_response"].sum()))

    greedy_curve = interpolate_curve(greedy_selected, budget_grid)
    random_curve = np.mean(random_curves, axis=0) if random_curves else np.zeros_like(budget_grid)
    curve_df = pd.DataFrame(
        {
            "budget_spend": budget_grid,
            "greedy_expected_incremental_response": greedy_curve,
            "random_expected_incremental_response": random_curve,
        }
    )

    summary = {
        "candidate_rows": int(len(candidates)),
        "budget": budget,
        "greedy_selected_rows": int(len(greedy_selected)),
        "greedy_expected_incremental_response": float(
            greedy_selected["expected_incremental_response"].sum()
        ),
        "random_expected_incremental_response_mean": float(np.mean(random_totals))
        if random_totals
        else 0.0,
        "random_expected_incremental_response_std": float(np.std(random_totals))
        if random_totals
        else 0.0,
    }
    return {"summary": summary, "curve": curve_df}


def plot_budget_curve(curve_df: pd.DataFrame) -> None:
    """Save greedy vs random cumulative value curve."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        curve_df["budget_spend"],
        curve_df["greedy_expected_incremental_response"],
        label="Greedy Uplift Strategy",
    )
    ax.plot(
        curve_df["budget_spend"],
        curve_df["random_expected_incremental_response"],
        linestyle="--",
        label="Random Baseline",
    )
    ax.set_title("Budget Allocation Simulation")
    ax.set_xlabel("Reward Budget Spent")
    ax.set_ylabel("Expected Incremental Purchases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(BUDGET_FIGURE_FILE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run greedy budget allocation against a random baseline."""
    scored_df = pd.read_csv(UPLIFT_SCORES_FILE)
    payload = run_budget_simulation(scored_df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload["curve"].to_csv(BUDGET_CURVE_FILE, index=False)
    with BUDGET_SUMMARY_FILE.open("w", encoding="utf-8") as handle:
        json.dump(payload["summary"], handle, indent=2)
    plot_budget_curve(payload["curve"])

    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
