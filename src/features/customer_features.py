"""Customer-level demographic and behavioral feature builders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROCESSED_DATA_DIR = Path("data/processed")
DEMOGRAPHIC_OUTPUT_FILE = PROCESSED_DATA_DIR / "demographic_features.parquet"
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


def main() -> None:
    """Load the merged response table and write demographic features to parquet."""
    response_merged_df = pd.read_parquet(PROCESSED_DATA_DIR / "response_merged.parquet")
    demographic_features = build_demographic_features(response_merged_df)
    DEMOGRAPHIC_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    demographic_features.to_parquet(DEMOGRAPHIC_OUTPUT_FILE, index=False)
    print(f"Saved {len(demographic_features)} rows to {DEMOGRAPHIC_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
