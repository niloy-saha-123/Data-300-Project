"""Assemble final modeling feature matrix."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROCESSED_DATA_DIR = Path("data/processed")
FEATURES_OUTPUT_FILE = PROCESSED_DATA_DIR / "features.parquet"
JOIN_KEYS = ["person", "offer_id", "received_time"]


def build_feature_matrix(
    response_df: pd.DataFrame,
    demographic_df: pd.DataFrame,
    behavioral_df: pd.DataFrame,
    offer_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join all feature tables into one modeling matrix."""
    features = response_df[JOIN_KEYS + ["label"]].copy()
    features = features.merge(
        demographic_df,
        on=JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )
    features = features.merge(
        behavioral_df,
        on=JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )
    features = features.merge(
        offer_df,
        on=JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )

    features["days_since_last_transaction"] = pd.to_numeric(
        features["days_since_last_transaction"], errors="coerce"
    )
    days_missing = features["days_since_last_transaction"].isna().astype(int)
    max_days = features["days_since_last_transaction"].max()
    fill_days = 0.0 if pd.isna(max_days) else float(max_days) + 1.0
    features["days_since_last_transaction_missing"] = days_missing
    features["days_since_last_transaction"] = features[
        "days_since_last_transaction"
    ].fillna(fill_days)

    features["income_to_difficulty_ratio"] = (
        features["income_imputed"] / features["difficulty"].where(features["difficulty"] != 0)
    ).fillna(0.0)

    features = features.sort_values(JOIN_KEYS, kind="stable").reset_index(drop=True)

    feature_columns = [column for column in features.columns if column != "label"]
    if features[feature_columns].isna().any().any():
        missing_columns = features[feature_columns].columns[
            features[feature_columns].isna().any()
        ].tolist()
        raise ValueError(f"Feature matrix still has missing values in {missing_columns}")

    return features


def main() -> None:
    """Load processed feature tables and write final modeling matrix."""
    response_df = pd.read_parquet(PROCESSED_DATA_DIR / "offer_response.parquet")
    demographic_df = pd.read_parquet(PROCESSED_DATA_DIR / "demographic_features.parquet")
    behavioral_df = pd.read_parquet(PROCESSED_DATA_DIR / "behavioral_features.parquet")
    offer_df = pd.read_parquet(PROCESSED_DATA_DIR / "offer_features.parquet")

    features = build_feature_matrix(
        response_df=response_df,
        demographic_df=demographic_df,
        behavioral_df=behavioral_df,
        offer_df=offer_df,
    )

    FEATURES_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_OUTPUT_FILE, index=False)
    print(f"Saved {len(features)} rows to {FEATURES_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
