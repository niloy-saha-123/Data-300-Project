"""Merge offer-response rows with customer and offer metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROCESSED_DATA_DIR = Path("data/processed")
MERGED_OUTPUT_FILE = PROCESSED_DATA_DIR / "response_merged.parquet"


def merge_response_data(
    response_df: pd.DataFrame, profile_df: pd.DataFrame, portfolio_df: pd.DataFrame
) -> pd.DataFrame:
    """Join response rows to customer demographics and offer attributes."""
    offer_features = portfolio_df.drop(columns=["offer_type", "duration"], errors="ignore")

    merged = response_df.merge(
        profile_df,
        on="person",
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        offer_features,
        on="offer_id",
        how="left",
        validate="many_to_one",
    )
    merged = merged.sort_values(
        ["person", "received_time", "offer_id"], kind="stable"
    ).reset_index(drop=True)

    return merged


def main() -> None:
    """Load processed inputs and write the merged response table to parquet."""
    response_df = pd.read_parquet(PROCESSED_DATA_DIR / "offer_response.parquet")
    profile_df = pd.read_parquet(PROCESSED_DATA_DIR / "profile_clean.parquet")
    portfolio_df = pd.read_parquet(PROCESSED_DATA_DIR / "portfolio_clean.parquet")

    merged = merge_response_data(response_df, profile_df, portfolio_df)
    MERGED_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(MERGED_OUTPUT_FILE, index=False)
    print(f"Saved {len(merged)} rows to {MERGED_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
