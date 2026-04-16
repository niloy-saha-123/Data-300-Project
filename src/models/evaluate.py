"""Evaluation helpers and report figure generation for trained models."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import joblib

PROJECT_ROOT = Path.cwd()
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.models.train import FEATURES_FILE, MODELS_DIR, prepare_xy, split_feature_matrix
from src.utils import plotting as _plotting  # noqa: F401


REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_OUTPUT_FILE = REPORTS_DIR / "validation_model_metrics.csv"
DEFAULT_MODEL_FILES = {
    "logistic_regression": MODELS_DIR / "logistic_regression.joblib",
    "random_forest": MODELS_DIR / "random_forest.joblib",
    "xgboost": MODELS_DIR / "xgboost.joblib",
}


def get_positive_scores(model: object, X: pd.DataFrame) -> pd.Series:
    """Return positive-class scores for a fitted classifier."""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
        scores = 1.0 / (1.0 + np.exp(-raw_scores))  # pragma: no cover
    else:  # pragma: no cover - unsupported model type
        raise ValueError("Model must implement predict_proba or decision_function")
    return pd.Series(scores, index=X.index, dtype=float)


def plot_roc_curve(
    model: object, X_test: pd.DataFrame, y_test: pd.Series, label: str, ax: plt.Axes
) -> float:
    """Add one ROC curve to an axis and return ROC-AUC."""
    scores = get_positive_scores(model, X_test)
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = float(roc_auc_score(y_test, scores))
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    return auc


def plot_precision_recall(
    model: object, X_test: pd.DataFrame, y_test: pd.Series, label: str, ax: plt.Axes
) -> float:
    """Add one precision-recall curve to an axis and return average precision."""
    scores = get_positive_scores(model, X_test)
    precision, recall, _ = precision_recall_curve(y_test, scores)
    average_precision = float(average_precision_score(y_test, scores))
    ax.plot(recall, precision, label=f"{label} (AP={average_precision:.3f})")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    return average_precision


def classification_report_df(
    model: object, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Return sklearn classification report as tidy DataFrame."""
    predictions = pd.Series(model.predict(X_test), index=X_test.index, dtype=int)
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    return report_df


def evaluate_model(
    model: object, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
) -> dict[str, float | str]:
    """Compute core validation metrics for one model."""
    scores = get_positive_scores(model, X_test)
    predictions = (scores >= 0.5).astype(int)
    return {
        "model": model_name,
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "average_precision": float(average_precision_score(y_test, scores)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
    }


def plot_confusion_matrix(
    model: object, X_test: pd.DataFrame, y_test: pd.Series, label: str, ax: plt.Axes
) -> None:
    """Plot one confusion matrix for a fitted model."""
    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    display.plot(ax=ax, colorbar=False)
    ax.set_title(label)


def load_available_models() -> dict[str, object]:
    """Load saved joblib models that exist on disk.

    Skips artifacts that fail to load (e.g. XGBoost model without ``xgboost``
    installed, or sklearn version mismatch).
    """
    models = {}
    for model_name, model_path in DEFAULT_MODEL_FILES.items():
        if not model_path.exists():
            continue
        try:
            models[model_name] = joblib.load(model_path)
        except (ModuleNotFoundError, ImportError, AttributeError, OSError) as exc:
            warnings.warn(
                f"Skipping model {model_name!r} at {model_path}: {exc}",
                UserWarning,
                stacklevel=1,
            )
    if not models:
        raise FileNotFoundError("No saved models found in models/")
    return models


def main() -> None:
    """Evaluate saved models on validation split and export figures."""
    features_df = pd.read_parquet(FEATURES_FILE)
    _, validation_df, _ = split_feature_matrix(features_df, strategy="auto")
    X_validation, y_validation = prepare_xy(validation_df)
    models = load_available_models()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    roc_fig, roc_ax = plt.subplots(figsize=(7, 5))
    pr_fig, pr_ax = plt.subplots(figsize=(7, 5))
    confusion_fig, confusion_axes = plt.subplots(
        1, len(models), figsize=(5 * len(models), 4)
    )
    if len(models) == 1:
        confusion_axes = [confusion_axes]

    metrics_rows = []
    report_frames = []

    for axis, (model_name, model) in zip(confusion_axes, models.items()):
        plot_roc_curve(model, X_validation, y_validation, model_name, roc_ax)
        plot_precision_recall(model, X_validation, y_validation, model_name, pr_ax)
        plot_confusion_matrix(model, X_validation, y_validation, model_name, axis)
        metrics_rows.append(evaluate_model(model, X_validation, y_validation, model_name))

        report_df = classification_report_df(model, X_validation, y_validation)
        report_df.insert(0, "model", model_name)
        report_frames.append(report_df)

    roc_ax.legend()
    pr_ax.legend()
    roc_fig.tight_layout()
    pr_fig.tight_layout()
    confusion_fig.tight_layout()

    roc_fig.savefig(FIGURES_DIR / "roc_curves.png", bbox_inches="tight")
    pr_fig.savefig(FIGURES_DIR / "precision_recall_curves.png", bbox_inches="tight")
    confusion_fig.savefig(FIGURES_DIR / "confusion_matrices.png", bbox_inches="tight")
    plt.close(roc_fig)
    plt.close(pr_fig)
    plt.close(confusion_fig)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(METRICS_OUTPUT_FILE, index=False)
    report_output = REPORTS_DIR / "validation_classification_reports.csv"
    pd.concat(report_frames, ignore_index=True).to_csv(report_output, index=False)

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
