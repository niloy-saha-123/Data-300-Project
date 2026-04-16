"""Customer segmentation with K-Means clustering."""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path.cwd()
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.models.train import FEATURES_FILE
from src.utils import plotting as _plotting  # noqa: F401


PROCESSED_DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
CUSTOMER_SEGMENTS_FILE = PROCESSED_DATA_DIR / "customer_segments.parquet"
CLUSTER_EVAL_FILE = REPORTS_DIR / "cluster_model_selection.csv"
CLUSTER_PROFILE_FILE = REPORTS_DIR / "cluster_profiles.csv"
ELBOW_FIGURE_FILE = FIGURES_DIR / "kmeans_elbow.png"
SILHOUETTE_FIGURE_FILE = FIGURES_DIR / "kmeans_silhouette.png"
HEATMAP_FIGURE_FILE = FIGURES_DIR / "cluster_offer_response_heatmap.png"
CLUSTER_FEATURES = [
    "age_imputed",
    "income_imputed",
    "membership_duration_days",
    "n_transactions_before",
    "avg_spend_before",
    "offer_completion_rate_before",
]
CLUSTER_RANGE = range(2, 9)
RANDOM_STATE = 42


def build_customer_clustering_frame(features_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce offer-level feature matrix to one latest clustering row per customer."""
    latest_rows = (
        features_df.sort_values(["person", "received_time"], kind="stable")
        .groupby("person", as_index=False)
        .tail(1)
        .copy()
    )
    customer_df = latest_rows[["person"] + CLUSTER_FEATURES].reset_index(drop=True)
    return customer_df


def evaluate_kmeans_candidates(
    customer_df: pd.DataFrame, cluster_range: range = CLUSTER_RANGE
) -> pd.DataFrame:
    """Fit candidate K values and return inertia and silhouette metrics."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customer_df[CLUSTER_FEATURES])

    rows = []
    for n_clusters in cluster_range:
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X_scaled)
        rows.append(
            {
                "n_clusters": n_clusters,
                "inertia": float(model.inertia_),
                "silhouette_score": float(silhouette_score(X_scaled, labels)),
            }
        )

    return pd.DataFrame(rows)


def select_best_k(evaluation_df: pd.DataFrame) -> int:
    """Pick K with best silhouette score."""
    best_row = evaluation_df.sort_values(
        ["silhouette_score", "n_clusters"], ascending=[False, True]
    ).iloc[0]
    return int(best_row["n_clusters"])


def fit_customer_segments(customer_df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Fit final K-Means model and return customer segment assignments."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customer_df[CLUSTER_FEATURES])
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    labels = model.fit_predict(X_scaled)
    segments = customer_df.copy()
    segments["cluster"] = labels.astype(int)
    return segments.sort_values(["cluster", "person"], kind="stable").reset_index(drop=True)


def summarize_cluster_profiles(
    customer_segments_df: pd.DataFrame, features_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build cluster profiles and cluster-offer completion heatmap table."""
    profile_df = (
        customer_segments_df.groupby("cluster", as_index=False)[CLUSTER_FEATURES]
        .mean()
        .sort_values("cluster")
        .reset_index(drop=True)
    )

    offer_frame = features_df[["person", "label", "offer_type_bogo", "offer_type_discount"]].copy()
    offer_frame["offer_type"] = offer_frame["offer_type_bogo"].map(
        lambda value: "bogo" if value == 1 else "discount"
    )
    offer_frame = offer_frame.merge(
        customer_segments_df[["person", "cluster"]],
        on="person",
        how="left",
        validate="many_to_one",
    )
    heatmap_df = (
        offer_frame.groupby(["cluster", "offer_type"], as_index=False)["label"]
        .mean()
        .rename(columns={"label": "completion_rate"})
    )

    return profile_df, heatmap_df


def plot_cluster_selection(evaluation_df: pd.DataFrame) -> None:
    """Save elbow and silhouette plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(evaluation_df["n_clusters"], evaluation_df["inertia"], marker="o")
    ax.set_title("K-Means Elbow Curve")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    fig.tight_layout()
    fig.savefig(ELBOW_FIGURE_FILE, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(evaluation_df["n_clusters"], evaluation_df["silhouette_score"], marker="o")
    ax.set_title("K-Means Silhouette Scores")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette Score")
    fig.tight_layout()
    fig.savefig(SILHOUETTE_FIGURE_FILE, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_heatmap(heatmap_df: pd.DataFrame) -> None:
    """Save cluster vs offer-type completion heatmap."""
    pivot = heatmap_df.pivot(index="cluster", columns="offer_type", values="completion_rate")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    image = ax.imshow(pivot.values, aspect="auto")
    ax.set_title("Cluster Completion Rate by Offer Type")
    ax.set_xlabel("Offer Type")
    ax.set_ylabel("Cluster")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for row_index, cluster in enumerate(pivot.index):
        for col_index, offer_type in enumerate(pivot.columns):
            value = pivot.loc[cluster, offer_type]
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Completion Rate")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(HEATMAP_FIGURE_FILE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run K-Means model selection, fit final segments, and export outputs."""
    features_df = pd.read_parquet(FEATURES_FILE)
    customer_df = build_customer_clustering_frame(features_df)
    evaluation_df = evaluate_kmeans_candidates(customer_df)
    best_k = select_best_k(evaluation_df)
    customer_segments_df = fit_customer_segments(customer_df, best_k)
    profile_df, heatmap_df = summarize_cluster_profiles(customer_segments_df, features_df)

    CUSTOMER_SEGMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    customer_segments_df.to_parquet(CUSTOMER_SEGMENTS_FILE, index=False)
    evaluation_df.to_csv(CLUSTER_EVAL_FILE, index=False)
    profile_df.to_csv(CLUSTER_PROFILE_FILE, index=False)

    plot_cluster_selection(evaluation_df)
    plot_cluster_heatmap(heatmap_df)

    print(
        json.dumps(
            {
                "best_k": best_k,
                "customer_rows": int(len(customer_segments_df)),
                "profile_rows": int(len(profile_df)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
