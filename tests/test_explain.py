"""Tests for model explainability helpers."""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.models.explain import (
    choose_best_model,
    get_estimator_for_explanation,
    get_feature_importance_df,
    supports_tree_shap,
)


def make_feature_names() -> list[str]:
    """Return compact feature name fixture."""
    return ["f1", "f2", "f3", "f4"]


def test_choose_best_model_uses_validation_metric() -> None:
    metrics = {
        "logistic_regression": {"roc_auc": 0.74},
        "random_forest": {"roc_auc": 0.77},
        "xgboost": {"roc_auc": 0.79},
    }

    best_model = choose_best_model(metrics)

    assert best_model == "xgboost"


def test_get_feature_importance_df_handles_logistic_pipeline() -> None:
    X, y = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    model = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X, y)

    importance_df = get_feature_importance_df(model, make_feature_names())

    assert importance_df.columns.tolist() == ["feature", "importance"]
    assert len(importance_df) == 4
    assert importance_df["importance"].ge(0).all()
    assert isinstance(get_estimator_for_explanation(model), LogisticRegression)


def test_get_feature_importance_df_handles_tree_model() -> None:
    X, y = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    importance_df = get_feature_importance_df(model, make_feature_names())

    assert len(importance_df) == 4
    assert importance_df["importance"].sum() > 0


def test_supports_tree_shap_matches_estimator_type() -> None:
    X, y = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    tree_model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    linear_model = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    ).fit(X, y)

    assert supports_tree_shap(tree_model) is True
    assert supports_tree_shap(linear_model) is False
