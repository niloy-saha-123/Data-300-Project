"""Tests for model evaluation helpers."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.models.evaluate import (
    classification_report_df,
    evaluate_model,
    plot_precision_recall,
    plot_roc_curve,
)


def make_fitted_model() -> tuple[LogisticRegression, pd.DataFrame, pd.Series]:
    """Create a small fitted binary classifier fixture."""
    X, y = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    y_series = pd.Series(y)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_df, y_series)
    return model, X_df, y_series


def test_plot_roc_curve_adds_line_and_returns_auc() -> None:
    model, X_test, y_test = make_fitted_model()
    fig, ax = plt.subplots()

    auc = plot_roc_curve(model, X_test, y_test, "logreg", ax)

    assert 0.0 <= auc <= 1.0
    assert len(ax.lines) >= 1
    plt.close(fig)


def test_plot_precision_recall_adds_line_and_returns_average_precision() -> None:
    model, X_test, y_test = make_fitted_model()
    fig, ax = plt.subplots()

    average_precision = plot_precision_recall(model, X_test, y_test, "logreg", ax)

    assert 0.0 <= average_precision <= 1.0
    assert len(ax.lines) >= 1
    plt.close(fig)


def test_classification_report_df_returns_expected_columns() -> None:
    model, X_test, y_test = make_fitted_model()

    report_df = classification_report_df(model, X_test, y_test)

    assert {"label", "precision", "recall", "f1-score", "support"} <= set(
        report_df.columns
    )
    assert {"0", "1"} <= set(report_df["label"].astype(str))


def test_evaluate_model_returns_metric_summary() -> None:
    model, X_test, y_test = make_fitted_model()

    metrics = evaluate_model(model, X_test, y_test, "logreg")

    assert metrics["model"] == "logreg"
    for key in ["roc_auc", "average_precision", "f1", "precision", "recall"]:
        assert 0.0 <= float(metrics[key]) <= 1.0
