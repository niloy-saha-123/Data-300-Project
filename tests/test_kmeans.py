"""Tests for customer clustering helpers."""

import pandas as pd

from src.clustering.kmeans import (
    build_customer_clustering_frame,
    evaluate_kmeans_candidates,
    fit_customer_segments,
    select_best_k,
    summarize_cluster_profiles,
)


def make_features_df() -> pd.DataFrame:
    """Create compact clustering fixture with repeated customers."""
    rows = []
    for index in range(12):
        person = f"p{index // 2}"
        rows.append(
            {
                "person": person,
                "received_time": index,
                "age_imputed": 25 + index,
                "income_imputed": 40000 + (index * 1000),
                "membership_duration_days": 100 + index,
                "n_transactions_before": index % 5,
                "avg_spend_before": 10.0 + index,
                "offer_completion_rate_before": (index % 3) / 2,
                "offer_type_bogo": int(index % 2 == 0),
                "offer_type_discount": int(index % 2 == 1),
                "label": int(index % 2 == 0),
            }
        )
    return pd.DataFrame(rows)


def test_build_customer_clustering_frame_keeps_latest_row_per_person() -> None:
    features_df = make_features_df()

    customer_df = build_customer_clustering_frame(features_df)

    assert len(customer_df) == features_df["person"].nunique()
    assert customer_df["person"].is_unique


def test_evaluate_kmeans_candidates_returns_scores() -> None:
    customer_df = build_customer_clustering_frame(make_features_df())

    evaluation_df = evaluate_kmeans_candidates(customer_df, range(2, 4))

    assert evaluation_df["n_clusters"].tolist() == [2, 3]
    assert {"inertia", "silhouette_score"} <= set(evaluation_df.columns)
    assert select_best_k(evaluation_df) in {2, 3}


def test_fit_customer_segments_assigns_cluster_labels() -> None:
    customer_df = build_customer_clustering_frame(make_features_df())

    segments_df = fit_customer_segments(customer_df, n_clusters=2)

    assert "cluster" in segments_df.columns
    assert set(segments_df["cluster"].unique()) <= {0, 1}


def test_summarize_cluster_profiles_returns_profile_and_heatmap_tables() -> None:
    features_df = make_features_df()
    customer_df = build_customer_clustering_frame(features_df)
    segments_df = fit_customer_segments(customer_df, n_clusters=2)

    profile_df, heatmap_df = summarize_cluster_profiles(segments_df, features_df)

    assert "cluster" in profile_df.columns
    assert {"cluster", "offer_type", "completion_rate"} <= set(heatmap_df.columns)
