"""Offer-level feature builders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROCESSED_DATA_DIR = Path("data/processed")
OFFER_OUTPUT_FILE = PROCESSED_DATA_DIR / "offer_features.parquet"


def build_offer_features(response_merged_df: pd.DataFrame) -> pd.DataFrame:
    """Build offer-level features for each response row."""
    features = response_merged_df[
        [
            "person",
            "offer_id",
            "received_time",
            "offer_type",
            "difficulty",
            "reward",
            "duration_days",
            "channel_web",
            "channel_email",
            "channel_mobile",
            "channel_social",
        ]
    ].copy()

    features["offer_type_bogo"] = (features["offer_type"] == "bogo").astype(int)
    features["offer_type_discount"] = (features["offer_type"] == "discount").astype(int)
    features["reward_to_difficulty_ratio"] = (
        features["reward"] / features["difficulty"].where(features["difficulty"] != 0)
    ).fillna(0.0)

    offer_features = features[
        [
            "person",
            "offer_id",
            "received_time",
            "offer_type_bogo",
            "offer_type_discount",
            "difficulty",
            "reward",
            "duration_days",
            "channel_web",
            "channel_email",
            "channel_mobile",
            "channel_social",
            "reward_to_difficulty_ratio",
        ]
    ].sort_values(["person", "received_time", "offer_id"], kind="stable").reset_index(
        drop=True
    )

    return offer_features


def main() -> None:
    """Load the merged response table and write offer features to parquet."""
    response_merged_df = pd.read_parquet(PROCESSED_DATA_DIR / "response_merged.parquet")
    offer_features = build_offer_features(response_merged_df)
    OFFER_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    offer_features.to_parquet(OFFER_OUTPUT_FILE, index=False)
    print(f"Saved {len(offer_features)} rows to {OFFER_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
