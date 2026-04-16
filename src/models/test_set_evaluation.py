"""Evaluate saved models on the held-out test split (run after training)."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

PROJECT_ROOT = Path.cwd()
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.evaluate import get_positive_scores, load_available_models
from src.models.train import (
    FEATURES_FILE,
    choose_split_strategy,
    prepare_xy,
    split_feature_matrix,
)

REPORTS_DIR = Path("reports")
TEST_METRICS_JSON = REPORTS_DIR / "test_set_metrics.json"
TEST_METRICS_CSV = REPORTS_DIR / "test_set_model_metrics.csv"


def _safe_roc_auc(y_true: pd.Series, scores: pd.Series) -> float:
    """ROC-AUC when both classes are present; otherwise NaN with a warning."""
    if y_true.nunique() < 2:
        warnings.warn("Test set has a single class; ROC-AUC is undefined.", UserWarning)
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _safe_average_precision(y_true: pd.Series, scores: pd.Series) -> float:
    try:
        return float(average_precision_score(y_true, scores))
    except ValueError:
        return float("nan")


def _baseline_row(
    name: str,
    y_train: pd.Series,
    y_test: pd.Series,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, float | str]:
    """Simple baselines scored on the test split."""
    if name == "majority_class":
        majority = int(y_train.mode().iloc[0])
        scores = pd.Series(
            [float(majority)] * len(y_test), index=y_test.index, dtype=float
        )
    else:
        income_threshold = float(train_df["income_imputed"].median())
        scores = (
            (test_df["income_imputed"] > income_threshold)
            & (test_df["reward"] > test_df["difficulty"])
        ).astype(float)
        scores.index = y_test.index

    predictions = (scores >= 0.5).astype(int)
    return {
        "model": name,
        "roc_auc": _safe_roc_auc(y_test, scores),
        "average_precision": _safe_average_precision(y_test, scores),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
    }


def _model_row(
    model_name: str, model: object, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float | str]:
    scores = get_positive_scores(model, X_test)
    predictions = (scores >= 0.5).astype(int)
    return {
        "model": model_name,
        "roc_auc": _safe_roc_auc(y_test, scores),
        "average_precision": _safe_average_precision(y_test, scores),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
    }


def run_test_evaluation(
    features_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Load data, score test split, return metrics dict (and write reports)."""
    if features_df is None:
        features_df = pd.read_parquet(FEATURES_FILE)

    strategy = choose_split_strategy(features_df)
    train_df, _validation_df, test_df = split_feature_matrix(
        features_df, strategy="auto"
    )

    if test_df.empty:
        raise ValueError("Test split is empty; cannot compute test metrics.")

    X_train, y_train = prepare_xy(train_df)
    X_test, y_test = prepare_xy(test_df)

    payload: dict[str, object] = {
        "split": {
            "strategy": strategy,
            "test_rows": int(len(test_df)),
            "train_rows": int(len(train_df)),
        },
        "baselines": [
            _baseline_row("majority_class", y_train, y_test, train_df, test_df),
            _baseline_row("rule_based", y_train, y_test, train_df, test_df),
        ],
        "models": [],
    }

    models = load_available_models()
    payload["models"] = [
        _model_row(name, model, X_test, y_test) for name, model in models.items()
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with TEST_METRICS_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    baseline_df = pd.DataFrame(payload["baselines"])
    models_df = pd.DataFrame(payload["models"])
    out_df = pd.concat([baseline_df, models_df], ignore_index=True)
    column_order = [
        "model",
        "roc_auc",
        "average_precision",
        "f1",
        "precision",
        "recall",
    ]
    out_df = out_df[[c for c in column_order if c in out_df.columns]]
    out_df.to_csv(TEST_METRICS_CSV, index=False)

    return payload


def main() -> None:
    """Write test-set metrics for saved models and baselines."""
    payload = run_test_evaluation()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()