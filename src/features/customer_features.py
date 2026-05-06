"""Build customer features without peeking past offer receipt time."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROCESSED_DATA_DIR = Path("../../data/processed")
DEMOGRAPHIC_OUTPUT_FILE = PROCESSED_DATA_DIR / "demographic_features.parquet"
BEHAVIORAL_OUTPUT_FILE = PROCESSED_DATA_DIR / "behavioral_features.parquet"
GENDER_LEVELS = ["F", "M", "O", "unknown"]
OFFER_EVENTS = {"offer received", "offer viewed", "offer completed"}


def _fill_with_group_median(series: pd.Series) -> pd.Series:
    """Fill gaps with a within-group median when one exists."""
    non_null = series.dropna()
    if non_null.empty:
        return series
    return series.fillna(non_null.median())


def build_demographic_features(
    response_merged_df: pd.DataFrame, reference_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Attach stable customer attributes to each offer row."""
    features = response_merged_df[
        ["person", "offer_id", "received_time", "gender", "age", "income", "became_member_on"]
    ].copy()

    if reference_date is None:
        reference_date = pd.Timestamp(features["became_member_on"].max()).normalize()
    else:
        reference_date = pd.Timestamp(reference_date).normalize()

    features["age"] = pd.to_numeric(features["age"], errors="coerce")
    features["income"] = pd.to_numeric(features["income"], errors="coerce")
    features["gender_filled"] = features["gender"].fillna("unknown")

    overall_income_median = features["income"].median()
    features["income_imputed"] = features.groupby("gender_filled")["income"].transform(
        _fill_with_group_median
    )
    features["income_imputed"] = features["income_imputed"].fillna(overall_income_median)
    features["income_missing"] = features["income"].isna().astype(int)

    age_median = features["age"].median()
    features["age_imputed"] = features["age"].fillna(age_median)
    features["age_missing"] = features["age"].isna().astype(int)

    features["membership_duration_days"] = (
        reference_date - pd.to_datetime(features["became_member_on"]).dt.normalize()
    ).dt.days

    gender_dummies = pd.get_dummies(features["gender_filled"], prefix="gender")
    for level in GENDER_LEVELS:
        column = f"gender_{level}"
        if column not in gender_dummies.columns:
            gender_dummies[column] = 0

    demographic_features = pd.concat(
        [
            features[
                [
                    "person",
                    "offer_id",
                    "received_time",
                    "age_imputed",
                    "age_missing",
                    "income_imputed",
                    "income_missing",
                    "membership_duration_days",
                ]
            ],
            gender_dummies[[f"gender_{level}" for level in GENDER_LEVELS]].astype(int),
        ],
        axis=1,
    )

    return demographic_features.sort_values(
        ["person", "received_time", "offer_id"], kind="stable"
    ).reset_index(drop=True)


def _build_transaction_features(
    transcript_df: pd.DataFrame, response_base: pd.DataFrame
) -> pd.DataFrame:
    """Summarize spend history and recent velocity before each offer receipt."""
    transactions = transcript_df.loc[
        transcript_df["event"] == "transaction", ["person", "time", "amount"]
    ].copy()
    transactions["amount"] = pd.to_numeric(transactions["amount"], errors="coerce").fillna(0.0)
    transactions = transactions.sort_values(["person", "time"], kind="stable").reset_index(drop=True)

    if transactions.empty:
        features = response_base[["person", "offer_id", "received_time", "_row_id"]].copy()
        features["n_transactions_before"] = 0
        features["total_spend_before"] = 0.0
        features["avg_spend_before"] = 0.0
        features["days_since_last_transaction"] = pd.NA
        features["spend_last_7d"] = 0.0
        features["trans_last_7d"] = 0
        return features

    transactions["n_transactions_before"] = transactions.groupby("person").cumcount() + 1
    transactions["total_spend_before"] = transactions.groupby("person")["amount"].cumsum()
    transactions["avg_spend_before"] = (
        transactions["total_spend_before"] / transactions["n_transactions_before"]
    )
    transactions["last_transaction_time"] = transactions["time"]

    # 1. Base merge (Current state just before received_time)
    features = pd.merge_asof(
        response_base.sort_values(["received_time", "person"], kind="stable"),
        transactions[
            [
                "person",
                "time",
                "n_transactions_before",
                "total_spend_before",
                "avg_spend_before",
                "last_transaction_time",
            ]
        ].sort_values(["time", "person"], kind="stable"),
        left_on="received_time",
        right_on="time",
        by="person",
        direction="backward",
        allow_exact_matches=False,
    )

    # 2. Velocity merge (State exactly 7 days ago. Starbucks time is in hours, so 7 * 24 = 168)
    features["time_7d_ago"] = features["received_time"] - 168
    
    history_7d = pd.merge_asof(
        features.sort_values(["time_7d_ago", "person"], kind="stable"),
        transactions[
            ["person", "time", "n_transactions_before", "total_spend_before"]
        ].rename(columns={
            "n_transactions_before": "n_trans_7d_ago",
            "total_spend_before": "spend_7d_ago"
        }).sort_values(["time", "person"], kind="stable"),
        left_on="time_7d_ago",
        right_on="time",
        by="person",
        direction="backward",
        allow_exact_matches=True,
    )

    # Rejoin the 7-day history safely using _row_id
    features = features.merge(
        history_7d[["_row_id", "n_trans_7d_ago", "spend_7d_ago"]],
        on="_row_id",
        how="left"
    )

    # Fill NaNs for baseline features
    features["n_transactions_before"] = features["n_transactions_before"].fillna(0).astype(int)
    features["total_spend_before"] = features["total_spend_before"].fillna(0.0)
    features["avg_spend_before"] = features["avg_spend_before"].fillna(0.0)
    features["days_since_last_transaction"] = (
        features["received_time"] - features["last_transaction_time"]
    ) / 24.0

    # Fill NaNs for the 7-day history (NaN means 0 spend/trans before that point)
    features["n_trans_7d_ago"] = features["n_trans_7d_ago"].fillna(0).astype(int)
    features["spend_7d_ago"] = features["spend_7d_ago"].fillna(0.0)

    # 3. Calculate 7-day Velocity
    features["spend_last_7d"] = features["total_spend_before"] - features["spend_7d_ago"]
    features["trans_last_7d"] = features["n_transactions_before"] - features["n_trans_7d_ago"]

    # Cleanup
    features = features.drop(
        columns=["time", "last_transaction_time", "time_7d_ago", "n_trans_7d_ago", "spend_7d_ago"]
    )

    return features.sort_values(["person", "received_time", "offer_id"], kind="stable")


def _build_offer_history_features(
    transcript_df: pd.DataFrame, response_base: pd.DataFrame
) -> pd.DataFrame:
    """Count prior offer events using only timestamps already observed."""
    offer_history = transcript_df.loc[
        transcript_df["event"].isin(OFFER_EVENTS), ["person", "time", "event"]
    ].copy()

    if offer_history.empty:
        history = response_base[["person", "offer_id", "received_time", "_row_id"]].copy()
        history["offers_received_before"] = 0
        history["offers_viewed_before"] = 0
        history["offers_completed_before"] = 0
    else:
        offer_history = offer_history.sort_values(
            ["person", "time", "event"], kind="stable"
        ).reset_index(drop=True)
        offer_history["received_event"] = (offer_history["event"] == "offer received").astype(
            int
        )
        offer_history["viewed_event"] = (offer_history["event"] == "offer viewed").astype(int)
        offer_history["completed_event"] = (
            offer_history["event"] == "offer completed"
        ).astype(int)
        offer_history["offers_received_before"] = offer_history.groupby("person")[
            "received_event"
        ].cumsum()
        offer_history["offers_viewed_before"] = offer_history.groupby("person")[
            "viewed_event"
        ].cumsum()
        offer_history["offers_completed_before"] = offer_history.groupby("person")[
            "completed_event"
        ].cumsum()

        history = pd.merge_asof(
            response_base.sort_values(["received_time", "person"], kind="stable"),
            offer_history[
                [
                    "person",
                    "time",
                    "offers_received_before",
                    "offers_viewed_before",
                    "offers_completed_before",
                ]
            ].sort_values(["time", "person"], kind="stable"),
            left_on="received_time",
            right_on="time",
            by="person",
            direction="backward",
            allow_exact_matches=False,
        ).drop(columns=["time"])
        for column in [
            "offers_received_before",
            "offers_viewed_before",
            "offers_completed_before",
        ]:
            history[column] = history[column].fillna(0).astype(int)

    history["offer_view_rate_before"] = (
        history["offers_viewed_before"]
        / history["offers_received_before"].where(history["offers_received_before"] > 0)
    ).fillna(0.0)
    history["offer_completion_rate_before"] = (
        history["offers_completed_before"]
        / history["offers_received_before"].where(history["offers_received_before"] > 0)
    ).fillna(0.0)
    return history


def build_behavioral_features(
    transcript_df: pd.DataFrame, response_df: pd.DataFrame
) -> pd.DataFrame:
    """Roll up pre-receipt behavior for each modeling row."""
    response_base = response_df[["person", "offer_id", "received_time"]].copy()
    response_base = response_base.sort_values(
        ["person", "received_time", "offer_id"], kind="stable"
    ).reset_index(drop=True)
    response_base["_row_id"] = range(len(response_base))

    transaction_features = _build_transaction_features(transcript_df, response_base)
    history_features = _build_offer_history_features(transcript_df, response_base)

    behavioral_features = transaction_features.merge(
        history_features[
            [
                "_row_id",
                "offers_received_before",
                "offers_viewed_before",
                "offers_completed_before",
                "offer_view_rate_before",
                "offer_completion_rate_before",
            ]
        ],
        on="_row_id",
        how="left",
        validate="one_to_one",
    )

    return behavioral_features[
        [
            "person",
            "offer_id",
            "received_time",
            "n_transactions_before",
            "total_spend_before",
            "avg_spend_before",
            "days_since_last_transaction",
            "spend_last_7d",          
            "trans_last_7d",
            "offers_received_before",
            "offers_viewed_before",
            "offers_completed_before",
            "offer_view_rate_before",
            "offer_completion_rate_before",
        ]
    ].sort_values(["person", "received_time", "offer_id"], kind="stable").reset_index(
        drop=True
    )


def main() -> None:
    """Build customer features from merged receipts and event history."""
    response_merged_df = pd.read_parquet(PROCESSED_DATA_DIR / "response_merged.parquet")
    transcript_df = pd.read_parquet(PROCESSED_DATA_DIR / "transcript_flat.parquet")

    reference_date = (
        pd.Timestamp(response_merged_df["became_member_on"].max()).normalize()
        + pd.to_timedelta(float(transcript_df["time"].max()) / 24.0, unit="D")
    )
    demographic_features = build_demographic_features(
        response_merged_df, reference_date=reference_date
    )
    behavioral_features = build_behavioral_features(transcript_df, response_merged_df)

    DEMOGRAPHIC_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    demographic_features.to_parquet(DEMOGRAPHIC_OUTPUT_FILE, index=False)
    behavioral_features.to_parquet(BEHAVIORAL_OUTPUT_FILE, index=False)

    print(f"Saved {len(demographic_features)} rows to {DEMOGRAPHIC_OUTPUT_FILE}")
    print(f"Saved {len(behavioral_features)} rows to {BEHAVIORAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
