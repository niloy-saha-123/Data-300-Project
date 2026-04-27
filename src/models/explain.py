"""Export feature importance views for linear and tree models."""

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
IMPORTANCE_DIR = REPORTS_DIR / "feature_importance"
SHAP_SUMMARY_FILE = FIGURES_DIR / "shap_summary.png"
MODEL_FILES = {
    "logistic_regression": MODELS_DIR / "logistic_regression.joblib",
    "random_forest": MODELS_DIR / "random_forest.joblib",
    "xgboost": MODELS_DIR / "xgboost.joblib",
}
TREE_MODELS = ["random_forest", "xgboost"]
MAX_SHAP_ROWS = 2000


def load_model_metrics(path: Path = METRICS_FILE) -> dict:
    """Read saved validation metrics."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def choose_best_model(metrics: dict, metric_name: str = "roc_auc") -> str:
    """Pick best model among saved model entries."""
    available = [
        model_name
        for model_name in MODEL_FILES
        if model_name in metrics and metric_name in metrics[model_name]
    ]
    if not available:
        raise ValueError("No trained model metrics found for explanation.")
    return max(available, key=lambda model_name: metrics[model_name][metric_name])


def choose_best_tree_model(metrics: dict, metric_name: str = "roc_auc") -> str | None:
    """Pick best tree model for SHAP when one is available."""
    available = [
        model_name
        for model_name in TREE_MODELS
        if model_name in metrics and metric_name in metrics[model_name]
    ]
    if not available:
        return None
    return max(available, key=lambda model_name: metrics[model_name][metric_name])


def get_estimator_for_explanation(model: object) -> object:
    """Unwrap pipeline models so feature attribution hits real estimator."""
    if isinstance(model, Pipeline):
        return model.named_steps["classifier"]
    return model


def get_feature_importance_df(model: object, feature_names: list[str]) -> pd.DataFrame:
    """Return sorted feature importances for supported estimators."""
    estimator = get_estimator_for_explanation(model)

    if hasattr(estimator, "coef_"):
        raw_importance = pd.Series(estimator.coef_[0], index=feature_names)
        importance_df = pd.DataFrame(
            {
                "feature": raw_importance.index,
                "importance": raw_importance.abs().values,
                "signed_value": raw_importance.values,
            }
        )
    elif hasattr(estimator, "feature_importances_"):
        raw_importance = pd.Series(estimator.feature_importances_, index=feature_names)
        importance_df = pd.DataFrame(
            {
                "feature": raw_importance.index,
                "importance": raw_importance.values,
            }
        )
    else:  # pragma: no cover
        raise ValueError("Model does not expose coefficients or feature importances.")

    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str,
    output_path: Path,
    top_n: int = 15,
    signed_column: str | None = None,
) -> None:
    """Save a compact horizontal importance chart."""
    top_features = importance_df.head(top_n).copy()
    value_column = signed_column or "importance"
    top_features = top_features.sort_values(value_column, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = None
    if signed_column is not None:
        colors = ["#00704A" if value >= 0 else "#B22222" for value in top_features[value_column]]
    ax.barh(top_features["feature"], top_features[value_column], color=colors)
    ax.set_title(title)
    ax.set_xlabel("Coefficient" if signed_column else "Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def supports_tree_shap(model: object) -> bool:
    """Return true when TreeExplainer should work cleanly."""
    estimator = get_estimator_for_explanation(model)
    return hasattr(estimator, "feature_importances_")


def create_shap_summary_plot(model: object, X: pd.DataFrame, output_path: Path) -> bool:
    """Write one SHAP summary plot for a tree model."""
    if not supports_tree_shap(model):
        return False

    import shap

    estimator = get_estimator_for_explanation(model)
    sample = X.head(min(len(X), MAX_SHAP_ROWS)).copy()
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return True


def _export_model_artifacts(
    model_name: str, model: object, feature_names: list[str]
) -> dict[str, str]:
    """Save CSV and chart for one fitted model."""
    importance_df = get_feature_importance_df(model, feature_names)
    IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = IMPORTANCE_DIR / f"{model_name}_feature_importance.csv"
    png_path = FIGURES_DIR / f"{model_name}_feature_importance.png"
    importance_df.to_csv(csv_path, index=False)

    signed_column = "signed_value" if "signed_value" in importance_df.columns else None
    title = model_name.replace("_", " ").title()
    if model_name == "logistic_regression":
        title = "Logistic Regression Coefficients"
    elif model_name == "random_forest":
        title = "Random Forest Feature Importance"
    elif model_name == "xgboost":
        title = "XGBoost Feature Importance"

    plot_feature_importance(
        importance_df=importance_df,
        title=title,
        output_path=png_path,
        signed_column=signed_column,
    )
    return {"csv": str(csv_path), "figure": str(png_path)}


def main() -> None:
    """Export per-model importance tables and one SHAP summary."""
    metrics = load_model_metrics()
    features_df = pd.read_parquet(FEATURES_FILE)
    _, validation_df, _ = split_feature_matrix(features_df, strategy="auto")
    X_validation, _ = prepare_xy(validation_df)

    artifacts: dict[str, object] = {"models": {}}
    for model_name, model_path in MODEL_FILES.items():
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        artifacts["models"][model_name] = _export_model_artifacts(
            model_name=model_name,
            model=model,
            feature_names=X_validation.columns.tolist(),
        )

    best_model = choose_best_model(metrics)
    best_tree_model = choose_best_tree_model(metrics)
    shap_written = False
    if best_tree_model and MODEL_FILES[best_tree_model].exists():
        tree_model = joblib.load(MODEL_FILES[best_tree_model])
        shap_written = create_shap_summary_plot(tree_model, X_validation, SHAP_SUMMARY_FILE)

    artifacts["best_model"] = best_model
    artifacts["best_tree_model"] = best_tree_model
    artifacts["shap_summary_written"] = shap_written
    artifacts["shap_summary_file"] = str(SHAP_SUMMARY_FILE) if shap_written else None

    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
