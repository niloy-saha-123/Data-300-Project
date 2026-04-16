"""Feature importance and SHAP plots for trained models."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib

PROJECT_ROOT = Path.cwd()
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models.train import FEATURES_FILE, METRICS_FILE, prepare_xy, split_feature_matrix
from src.utils import plotting as _plotting  # noqa: F401


MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
IMPORTANCE_OUTPUT_FILE = REPORTS_DIR / "feature_importance.csv"
BAR_PLOT_FILE = FIGURES_DIR / "feature_importance_bar.png"
SHAP_SUMMARY_FILE = FIGURES_DIR / "shap_summary.png"
MODEL_FILES = {
    "logistic_regression": MODELS_DIR / "logistic_regression.joblib",
    "random_forest": MODELS_DIR / "random_forest.joblib",
    "xgboost": MODELS_DIR / "xgboost.joblib",
}
MODEL_CANDIDATES = ["logistic_regression", "random_forest", "xgboost"]
MAX_SHAP_ROWS = 2000


def load_model_metrics(path: Path = METRICS_FILE) -> dict:
    """Load saved training metrics JSON."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def choose_best_model(metrics: dict, metric_name: str = "roc_auc") -> str:
    """Pick best trained model by chosen validation metric."""
    available = [
        model_name
        for model_name in MODEL_CANDIDATES
        if model_name in metrics and metric_name in metrics[model_name]
    ]
    if not available:
        raise ValueError("No trained model metrics found for explanation")
    return max(available, key=lambda model_name: metrics[model_name][metric_name])


def get_estimator_for_explanation(model: object) -> object:
    """Return final estimator when model is wrapped in pipeline."""
    if isinstance(model, Pipeline):
        return model.named_steps["classifier"]
    return model


def get_feature_importance_df(model: object, feature_names: list[str]) -> pd.DataFrame:
    """Build a tidy feature-importance table for supported models."""
    estimator = get_estimator_for_explanation(model)

    if hasattr(estimator, "coef_"):
        importances = pd.Series(estimator.coef_[0], index=feature_names).abs()
    elif hasattr(estimator, "feature_importances_"):
        importances = pd.Series(estimator.feature_importances_, index=feature_names)
    else:  # pragma: no cover - unsupported estimator
        raise ValueError("Model does not expose coefficient or feature_importances_ values")

    importance_df = (
        importances.rename("importance")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame, title: str, output_path: Path, top_n: int = 15
) -> None:
    """Plot top feature importances as horizontal bar chart."""
    top_features = importance_df.head(top_n).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features["feature"], top_features["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def supports_tree_shap(model: object) -> bool:
    """Return true when estimator can use TreeExplainer cleanly."""
    estimator = get_estimator_for_explanation(model)
    return hasattr(estimator, "feature_importances_")


def create_shap_summary_plot(model: object, X: pd.DataFrame, output_path: Path) -> bool:
    """Create SHAP summary plot for tree-based models."""
    if not supports_tree_shap(model):
        return False

    import shap

    estimator = get_estimator_for_explanation(model)
    sample = X.head(min(len(X), MAX_SHAP_ROWS)).copy()
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(sample)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return True


def main() -> None:
    """Pick best model, export feature-importance plot, and save SHAP summary."""
    metrics = load_model_metrics()
    best_model_name = choose_best_model(metrics)
    model_path = MODEL_FILES[best_model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Missing saved model for {best_model_name}")

    model = joblib.load(model_path)
    features_df = pd.read_parquet(FEATURES_FILE)
    _, validation_df, _ = split_feature_matrix(features_df, strategy="auto")
    X_validation, _ = prepare_xy(validation_df)

    importance_df = get_feature_importance_df(model, X_validation.columns.tolist())
    IMPORTANCE_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(IMPORTANCE_OUTPUT_FILE, index=False)
    plot_feature_importance(
        importance_df,
        title=f"{best_model_name.replace('_', ' ').title()} Feature Importance",
        output_path=BAR_PLOT_FILE,
    )

    shap_written = create_shap_summary_plot(model, X_validation, SHAP_SUMMARY_FILE)
    print(
        json.dumps(
            {
                "best_model": best_model_name,
                "bar_plot": str(BAR_PLOT_FILE),
                "shap_summary_written": shap_written,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
