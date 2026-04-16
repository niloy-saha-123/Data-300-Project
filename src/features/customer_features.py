"""Customer-level demographic and behavioral feature builders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROCESSED_DATA_DIR = Path("data/processed")
DEMOGRAPHIC_OUTPUT_FILE = PROCESSED_DATA_DIR / "demographic_features.parquet"
BEHAVIORAL_OUTPUT_FILE = PROCESSED_DATA_DIR / "behavioral_features.parquet"
GENDER_LEVELS = ["F", "M", "O", "unknown"]


def _fill_with_group_median(series: pd.Series) -> pd.Series:
    """Fill missing numeric values with the non-null median within a group."""
    non_null = series.dropna()
    if non_null.empty:
        return series
    return series.fillna(non_null.median())


def build_demographic_features(
    response_merged_df: pd.DataFrame, reference_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Build demographic features for each offer-response row."""
    features = response_merged_df[
        ["person", "offer_id", "received_time", "gender", "age", "income", "became_member_on"]
    ].copy()

    if reference_date is None:
        # Transcript time is relative, so anchor tenure to the latest observed
        # calendar membership date in the dataset unless a reference date is supplied.
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

    demographic_features = demographic_features.sort_values(
        ["person", "received_time", "offer_id"], kind="stable"
    ).reset_index(drop=True)

    return demographic_features


def build_behavioral_features(
    transcript_df: pd.DataFrame, response_df: pd.DataFrame
) -> pd.DataFrame:
    """Build pre-offer behavioral features for each offer-response row."""
    response_base = response_df[
        ["person", "offer_id", "received_time", "label"]
    ].copy()
    response_base = response_base.sort_values(
        ["person", "received_time", "offer_id"], kind="stable"
    ).reset_index(drop=True)
    response_base["_row_id"] = range(len(response_base))

    transactions = transcript_df.loc[
        transcript_df["event"] == "transaction", ["person", "time", "amount"]
    ].copy()
    transactions["amount"] = pd.to_numeric(transactions["amount"], errors="coerce").fillna(
        0.0
    )
    transactions = transactions.sort_values(["person", "time"], kind="stable").reset_index(
        drop=True
    )

    if transactions.empty:
        transaction_features = response_base[["person", "offer_id", "received_time", "_row_id"]].copy()
        transaction_features["n_transactions_before"] = 0
        transaction_features["total_spend_before"] = 0.0
        transaction_features["avg_spend_before"] = 0.0
        transaction_features["days_since_last_transaction"] = pd.NA
    else:
        transactions["n_transactions_before"] = transactions.groupby("person").cumcount() + 1
        transactions["total_spend_before"] = transactions.groupby("person")["amount"].cumsum()
        transactions["avg_spend_before"] = (
            transactions["total_spend_before"] / transactions["n_transactions_before"]
        )
        transactions["last_transaction_time"] = transactions["time"]
        transactions_by_person = {
            person: group[
                [
                    "time",
                    "n_transactions_before",
                    "total_spend_before",
                    "avg_spend_before",
                    "last_transaction_time",
                ]
            ]
            .sort_values("time", kind="stable")
            .reset_index(drop=True)
            for person, group in transactions.groupby("person", sort=False)
        }

        transaction_feature_parts = []
        for person, person_rows in response_base.groupby("person", sort=False):
            left = person_rows[
                ["person", "offer_id", "received_time", "_row_id"]
            ].sort_values("received_time", kind="stable")
            person_transactions = transactions_by_person.get(person)

            if person_transactions is None or person_transactions.empty:
                merged = left.copy()
                merged["n_transactions_before"] = 0
                merged["total_spend_before"] = 0.0
                merged["avg_spend_before"] = 0.0
                merged["days_since_last_transaction"] = float("nan")
            else:
                merged = pd.merge_asof(
                    left,
                    person_transactions,
                    left_on="received_time",
                    right_on="time",
                    direction="backward",
                    allow_exact_matches=False,
                )
                merged["n_transactions_before"] = (
                    merged["n_transactions_before"].fillna(0).astype(int)
                )
                merged["total_spend_before"] = merged["total_spend_before"].fillna(0.0)
                merged["avg_spend_before"] = merged["avg_spend_before"].fillna(0.0)
                merged["days_since_last_transaction"] = (
                    merged["received_time"] - merged["last_transaction_time"]
                ) / 24.0
                merged = merged.drop(columns=["time", "last_transaction_time"])

            transaction_feature_parts.append(merged)

        transaction_features = pd.concat(
            transaction_feature_parts, ignore_index=True
        ).sort_values(["person", "received_time", "offer_id"], kind="stable")

    history_features = response_base[["person", "offer_id", "received_time", "_row_id"]].copy()
    history_parts = []
    for person, person_rows in response_base.groupby("person", sort=False):
        person_history = person_rows[
            ["person", "offer_id", "received_time", "_row_id"]
        ].copy()
        offers_received_before = []
        offers_completed_before = []
        prior_received = 0
        prior_completed = 0

        for _, time_rows in person_rows.groupby("received_time", sort=True):
            group_size = len(time_rows)
            group_completions = int(time_rows["label"].sum())
            offers_received_before.extend([prior_received] * group_size)
            offers_completed_before.extend([prior_completed] * group_size)
            prior_received += group_size
            prior_completed += group_completions

        person_history["offers_received_before"] = offers_received_before
        person_history["offers_completed_before"] = offers_completed_before
        history_parts.append(person_history)

    history_features = pd.concat(history_parts, ignore_index=True)
    history_features["offers_received_before"] = history_features["offers_received_before"].astype(
        int
    )
    history_features["offers_completed_before"] = history_features[
        "offers_completed_before"
    ].astype(int)
    history_features["offer_completion_rate_before"] = (
        history_features["offers_completed_before"]
        / history_features["offers_received_before"].where(
            history_features["offers_received_before"] > 0
        )
    ).fillna(0.0)

    behavioral_features = transaction_features.merge(
        history_features[
            [
                "_row_id",
                "offers_received_before",
                "offers_completed_before",
                "offer_completion_rate_before",
            ]
        ],
        on="_row_id",
        how="left",
        validate="one_to_one",
    )

    behavioral_features = behavioral_features[
        [
            "person",
            "offer_id",
            "received_time",
            "n_transactions_before",
            "total_spend_before",
            "avg_spend_before",
            "days_since_last_transaction",
            "offers_received_before",
            "offers_completed_before",
            "offer_completion_rate_before",
        ]
    ]
    behavioral_features = behavioral_features.sort_values(
        ["person", "received_time", "offer_id"], kind="stable"
    ).reset_index(drop=True)

    return behavioral_features


def main() -> None:
    """Load processed inputs and write customer feature tables to parquet."""
    response_merged_df = pd.read_parquet(PROCESSED_DATA_DIR / "response_merged.parquet")
    transcript_df = pd.read_parquet(PROCESSED_DATA_DIR / "transcript_flat.parquet")
    demographic_features = build_demographic_features(response_merged_df)
    behavioral_features = build_behavioral_features(transcript_df, response_merged_df)
    DEMOGRAPHIC_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    demographic_features.to_parquet(DEMOGRAPHIC_OUTPUT_FILE, index=False)
    behavioral_features.to_parquet(BEHAVIORAL_OUTPUT_FILE, index=False)
    print(f"Saved {len(demographic_features)} rows to {DEMOGRAPHIC_OUTPUT_FILE}")
    print(f"Saved {len(behavioral_features)} rows to {BEHAVIORAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
