"""Utilities for loading and cleaning raw Starbucks dataset files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
CHANNEL_COLUMNS = ["web", "email", "mobile", "social"]


def read_jsonl(path: str | Path) -> pd.DataFrame:
    """Read a newline-delimited JSON file while skipping blank lines."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            records.append(json.loads(raw))

    return pd.DataFrame.from_records(records)


def load_portfolio(path: str | Path) -> pd.DataFrame:
    """Load portfolio.json and expand channels into indicator columns."""
    portfolio = read_jsonl(path)
    portfolio = portfolio.rename(columns={"id": "offer_id"}).copy()

    for channel in CHANNEL_COLUMNS:
        portfolio[f"channel_{channel}"] = portfolio["channels"].apply(
            lambda values: int(channel in values)
        )

    portfolio["duration"] = portfolio["duration"].astype(int)
    portfolio = portfolio.drop(columns=["channels"]).sort_values("offer_id")
    portfolio = portfolio.reset_index(drop=True)

    return portfolio


def load_profile(path: str | Path) -> pd.DataFrame:
    """Load profile.json and clean membership date and missing demographics."""
    profile = read_jsonl(path)
    profile = profile.rename(columns={"id": "person"}).copy()

    profile["became_member_on"] = pd.to_datetime(
        profile["became_member_on"], format="%Y%m%d"
    )
    profile.loc[profile["age"] == 118, "age"] = pd.NA
    profile = profile.sort_values("person").reset_index(drop=True)

    return profile


def load_transcript(path: str | Path) -> pd.DataFrame:
    """Load transcript.json and flatten the nested value payload."""
    transcript = read_jsonl(path).copy()
    value_columns = pd.json_normalize(transcript["value"])

    if "offer id" in value_columns.columns:
        if "offer_id" in value_columns.columns:
            value_columns["offer_id"] = value_columns["offer_id"].fillna(
                value_columns["offer id"]
            )
        else:
            value_columns["offer_id"] = value_columns["offer id"]

    for column in ["offer_id", "amount", "reward"]:
        if column not in value_columns.columns:
            value_columns[column] = pd.NA

    transcript = pd.concat(
        [transcript.drop(columns=["value"]), value_columns[["offer_id", "amount", "reward"]]],
        axis=1,
    )

    return transcript


def main() -> None:
    """Load raw data files and write cleaned portfolio/profile tables to parquet."""
    portfolio_path = RAW_DATA_DIR / "portfolio.json"
    profile_path = RAW_DATA_DIR / "profile.json"
    portfolio_output_path = PROCESSED_DATA_DIR / "portfolio_clean.parquet"
    profile_output_path = PROCESSED_DATA_DIR / "profile_clean.parquet"

    portfolio = load_portfolio(portfolio_path)
    profile = load_profile(profile_path)

    portfolio_output_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio.to_parquet(portfolio_output_path, index=False)
    profile.to_parquet(profile_output_path, index=False)

    print(f"Saved {len(portfolio)} rows to {portfolio_output_path}")
    print(f"Saved {len(profile)} rows to {profile_output_path}")


if __name__ == "__main__":
    main()
