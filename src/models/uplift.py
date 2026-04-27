"""Estimate incremental offer impact with a simple T-learner."""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path.cwd()
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data.parse_raw import PROCESSED_DATA_DIR
from src.features.build_features import build_feature_matrix
from src.features.customer_features import (
    build_behavioral_features,
    build_demographic_features,
)
from src.features.offer_features import build_offer_features
from src.models.train import ID_COLUMNS, RANDOM_STATE
from src.utils import plotting as _plotting  # noqa: F401


MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
UPLIFT_FEATURES_FILE = PROCESSED_DATA_DIR / "uplift_features.parquet"
UPLIFT_SCORES_FILE = REPORTS_DIR / "uplift_test_scores.csv"
UPLIFT_CURVE_FILE = REPORTS_DIR / "uplift_curve.csv"
UPLIFT_METRICS_FILE = REPORTS_DIR / "uplift_metrics.json"
QINI_FIGURE_FILE = FIGURES_DIR / "uplift_qini.png"
GAIN_FIGURE_FILE = FIGURES_DIR / "uplift_cumulative_gain.png"
TREATMENT_MODEL_FILE = MODELS_DIR / "uplift_treatment_model.joblib"
CONTROL_MODEL_FILE = MODELS_DIR / "uplift_control_model.joblib"


def build_receipt_frame(
    transcript_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per offer receipt, including informational offers."""
    receipts = transcript_df.loc[
        transcript_df["event"] == "offer received", ["person", "offer_id", "time"]
    ].rename(columns={"time": "received_time"})
    receipts = receipts.merge(
        portfolio_df,
        on="offer_id",
        how="left",
        validate="many_to_one",
    )
    receipts["duration_days"] = receipts["duration"].astype(int)
    receipts["window_end_time"] = receipts["received_time"] + (receipts["duration_days"] * 24)
    receipts["purchase_within_window"] = label_purchase_window_outcome(
        receipts, transcript_df
    )
    receipts["treatment"] = receipts["offer_type"].isin(["bogo", "discount"]).astype(int)

    merged = receipts.merge(profile_df, on="person", how="left", validate="many_to_one")
    return merged.sort_values(["person", "received_time", "offer_id"], kind="stable").reset_index(
        drop=True
    )


def label_purchase_window_outcome(
    receipt_df: pd.DataFrame, transcript_df: pd.DataFrame
) -> pd.Series:
    """Flag whether any transaction lands inside the offer window."""
    transactions = transcript_df.loc[
        transcript_df["event"] == "transaction", ["person", "time"]
    ].sort_values(["person", "time"], kind="stable")
    transaction_times = {
        person: group["time"].to_numpy()
        for person, group in transactions.groupby("person", sort=False)
    }

    outcomes = np.zeros(len(receipt_df), dtype=int)
    for index, row in receipt_df.reset_index(drop=True).iterrows():
        person_times = transaction_times.get(row["person"])
        if person_times is None or len(person_times) == 0:
            continue
        left = np.searchsorted(person_times, row["received_time"], side="right")
        right = np.searchsorted(person_times, row["window_end_time"], side="right")
        outcomes[index] = int(right > left)
    return pd.Series(outcomes, index=receipt_df.index, dtype=int)


def build_uplift_feature_matrix(
    transcript_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble model-ready uplift dataset from receipt windows."""
    receipt_df = build_receipt_frame(transcript_df, profile_df, portfolio_df)
    reference_date = (
        pd.Timestamp(profile_df["became_member_on"].max()).normalize()
        + pd.to_timedelta(float(transcript_df["time"].max()) / 24.0, unit="D")
    )
    demographic_df = build_demographic_features(receipt_df, reference_date=reference_date)
    behavioral_df = build_behavioral_features(transcript_df, receipt_df)
    offer_df = build_offer_features(receipt_df)

    feature_input = receipt_df[["person", "offer_id", "received_time"]].copy()
    feature_input["label"] = receipt_df["purchase_within_window"].astype(int)
    features_df = build_feature_matrix(feature_input, demographic_df, behavioral_df, offer_df)
    features_df["treatment"] = receipt_df["treatment"].values
    features_df["offer_type"] = receipt_df["offer_type"].values
    return features_df


def split_uplift_dataset(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a stratified test set on treatment/outcome balance."""
    strata = (
        features_df["treatment"].astype(str) + "_" + features_df["label"].astype(str)
    )
    train_df, test_df = train_test_split(
        features_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=strata,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_uplift_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return feature columns used by treatment and control models."""
    return [
        column
        for column in features_df.columns
        if column not in ID_COLUMNS + ["label", "treatment", "offer_type"]
    ]


def fit_t_learner(train_df: pd.DataFrame) -> tuple[RandomForestClassifier, RandomForestClassifier]:
    """Fit separate outcome models for treated and control groups."""
    feature_columns = get_uplift_feature_columns(train_df)
    treated_df = train_df.loc[train_df["treatment"] == 1].copy()
    control_df = train_df.loc[train_df["treatment"] == 0].copy()

    treatment_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=25,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    control_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=25,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    treatment_model.fit(treated_df[feature_columns], treated_df["label"])
    control_model.fit(control_df[feature_columns], control_df["label"])
    return treatment_model, control_model


def score_uplift(
    treatment_model: RandomForestClassifier,
    control_model: RandomForestClassifier,
    scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict treatment uplift for each scored row."""
    feature_columns = get_uplift_feature_columns(scored_df)
    scored = scored_df[
        ID_COLUMNS + ["label", "treatment", "offer_type", "difficulty", "reward"]
    ].copy()
    scored["p_treatment"] = treatment_model.predict_proba(scored_df[feature_columns])[:, 1]
    scored["p_control"] = control_model.predict_proba(scored_df[feature_columns])[:, 1]
    scored["uplift_score"] = scored["p_treatment"] - scored["p_control"]
    return scored.sort_values("uplift_score", ascending=False).reset_index(drop=True)


def build_uplift_curve(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative gain and Qini-style curves from uplift ranking."""
    ranked = scored_df.sort_values("uplift_score", ascending=False).reset_index(drop=True)
    ranked["population_fraction"] = (np.arange(len(ranked)) + 1) / len(ranked)
    ranked["cum_treated"] = ranked["treatment"].cumsum()
    ranked["cum_control"] = (1 - ranked["treatment"]).cumsum()
    ranked["cum_treated_outcome"] = (ranked["treatment"] * ranked["label"]).cumsum()
    ranked["cum_control_outcome"] = ((1 - ranked["treatment"]) * ranked["label"]).cumsum()

    adjusted_control = ranked["cum_control_outcome"] * (
        ranked["cum_treated"] / ranked["cum_control"].replace(0, np.nan)
    )
    ranked["cumulative_gain"] = (
        ranked["cum_treated_outcome"] - adjusted_control.fillna(0.0)
    )
    ranked["random_baseline"] = (
        ranked["cumulative_gain"].iloc[-1] * ranked["population_fraction"]
    )
    ranked["qini"] = ranked["cumulative_gain"] - ranked["random_baseline"]
    return ranked


def plot_uplift_curves(curve_df: pd.DataFrame) -> None:
    """Write cumulative gain and Qini plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve_df["population_fraction"], curve_df["cumulative_gain"], label="T-Learner")
    ax.plot(
        curve_df["population_fraction"],
        curve_df["random_baseline"],
        linestyle="--",
        label="Random Baseline",
    )
    ax.set_title("Cumulative Gain Curve")
    ax.set_xlabel("Population Fraction Targeted")
    ax.set_ylabel("Incremental Purchases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GAIN_FIGURE_FILE, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve_df["population_fraction"], curve_df["qini"], label="Qini")
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1)
    ax.set_title("Qini Curve")
    ax.set_xlabel("Population Fraction Targeted")
    ax.set_ylabel("Incremental Gain Over Random")
    ax.legend()
    fig.tight_layout()
    fig.savefig(QINI_FIGURE_FILE, bbox_inches="tight")
    plt.close(fig)


def _safe_auc(y_true: pd.Series, scores: pd.Series) -> float:
    """Return ROC-AUC when both classes exist."""
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def main() -> None:
    """Fit uplift models, write scored test set, and export curves."""
    transcript_df = pd.read_parquet(PROCESSED_DATA_DIR / "transcript_flat.parquet")
    profile_df = pd.read_parquet(PROCESSED_DATA_DIR / "profile_clean.parquet")
    portfolio_df = pd.read_parquet(PROCESSED_DATA_DIR / "portfolio_clean.parquet")

    features_df = build_uplift_feature_matrix(transcript_df, profile_df, portfolio_df)
    features_df.to_parquet(UPLIFT_FEATURES_FILE, index=False)

    train_df, test_df = split_uplift_dataset(features_df)
    treatment_model, control_model = fit_t_learner(train_df)
    scored_df = score_uplift(treatment_model, control_model, test_df)
    curve_df = build_uplift_curve(scored_df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(treatment_model, TREATMENT_MODEL_FILE)
    joblib.dump(control_model, CONTROL_MODEL_FILE)
    scored_df.to_csv(UPLIFT_SCORES_FILE, index=False)
    curve_df.to_csv(UPLIFT_CURVE_FILE, index=False)
    plot_uplift_curves(curve_df)

    feature_columns = get_uplift_feature_columns(test_df)
    treated_test = test_df.loc[test_df["treatment"] == 1]
    control_test = test_df.loc[test_df["treatment"] == 0]
    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "treated_test_rows": int(len(treated_test)),
        "control_test_rows": int(len(control_test)),
        "treated_model_roc_auc": _safe_auc(
            treated_test["label"],
            pd.Series(
                treatment_model.predict_proba(treated_test[feature_columns])[:, 1],
                index=treated_test.index,
            ),
        ),
        "control_model_roc_auc": _safe_auc(
            control_test["label"],
            pd.Series(
                control_model.predict_proba(control_test[feature_columns])[:, 1],
                index=control_test.index,
            ),
        ),
        "qini_auc": float(
            np.trapezoid(curve_df["qini"], curve_df["population_fraction"])
        ),
        "top_30pct_incremental_gain": float(
            curve_df.loc[curve_df["population_fraction"] <= 0.3, "cumulative_gain"].iloc[-1]
        ),
    }
    with UPLIFT_METRICS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
