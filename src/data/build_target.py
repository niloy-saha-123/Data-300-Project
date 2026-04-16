"""Build the offer-response target table from parsed transcript events."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


PROCESSED_DATA_DIR = Path("data/processed")
OFFER_EVENTS = {"offer received", "offer viewed", "offer completed"}


def get_offer_events(transcript_df: pd.DataFrame) -> pd.DataFrame:
    """Filter the transcript to offer-related events with a non-null offer_id."""
    offer_events = transcript_df.loc[
        transcript_df["event"].isin(OFFER_EVENTS) & transcript_df["offer_id"].notna()
    ].copy()
    offer_events = offer_events.sort_values(["person", "time", "event"]).reset_index(
        drop=True
    )
    return offer_events


def build_response_table(
    offer_events_df: pd.DataFrame, portfolio_df: pd.DataFrame
) -> pd.DataFrame:
    """Build one response row per offer receipt using view-before-complete logic."""
    portfolio_lookup = (
        portfolio_df.loc[portfolio_df["offer_type"] != "informational"]
        .assign(duration_hours=lambda df: df["duration"] * 24)
        .set_index("offer_id")[["offer_type", "duration", "duration_hours"]]
        .to_dict("index")
    )

    receipts = []

    for person, person_events in offer_events_df.groupby("person", sort=False):
        active_receipts: dict[str, list[dict[str, object]]] = defaultdict(list)
        sorted_events = person_events.sort_values(["time", "event"], kind="stable")

        for event in sorted_events.itertuples(index=False):
            offer_id = event.offer_id
            if offer_id not in portfolio_lookup:
                continue

            offer_meta = portfolio_lookup[offer_id]
            event_time = int(event.time)

            if event.event == "offer received":
                receipt = {
                    "person": person,
                    "offer_id": offer_id,
                    "offer_type": offer_meta["offer_type"],
                    "received_time": event_time,
                    "window_end_time": event_time + int(offer_meta["duration_hours"]),
                    "duration_days": int(offer_meta["duration"]),
                    "viewed": False,
                    "viewed_time": pd.NA,
                    "completed": False,
                    "completed_time": pd.NA,
                    "completed_after_view": False,
                }
                receipts.append(receipt)
                active_receipts[offer_id].append(receipt)
                continue

            eligible_receipts = [
                receipt
                for receipt in active_receipts[offer_id]
                if receipt["received_time"] <= event_time <= receipt["window_end_time"]
            ]

            if event.event == "offer viewed":
                for receipt in eligible_receipts:
                    if not receipt["viewed"]:
                        receipt["viewed"] = True
                        receipt["viewed_time"] = event_time
                        break
                continue

            if event.event == "offer completed":
                for receipt in eligible_receipts:
                    viewed_time = receipt["viewed_time"]
                    if (
                        receipt["viewed"]
                        and not receipt["completed"]
                        and pd.notna(viewed_time)
                        and int(viewed_time) < event_time
                    ):
                        receipt["completed"] = True
                        receipt["completed_time"] = event_time
                        receipt["completed_after_view"] = True
                        break

    response_df = pd.DataFrame(receipts)
    if response_df.empty:
        return response_df

    response_df["label"] = response_df["completed_after_view"].astype(int)
    response_df = response_df.sort_values(
        ["person", "received_time", "offer_id"], kind="stable"
    ).reset_index(drop=True)

    return response_df


def main() -> None:
    """Load parsed inputs and write the offer response table to parquet."""
    transcript = pd.read_parquet(PROCESSED_DATA_DIR / "transcript_flat.parquet")
    portfolio = pd.read_parquet(PROCESSED_DATA_DIR / "portfolio_clean.parquet")

    offer_events = get_offer_events(transcript)
    response_df = build_response_table(offer_events, portfolio)

    output_path = PROCESSED_DATA_DIR / "offer_response.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response_df.to_parquet(output_path, index=False)
    print(f"Saved {len(response_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
