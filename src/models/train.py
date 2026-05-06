"""Train reproducible offer-response models and baseline comparators."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - depends on local environment
    XGBClassifier = None

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*encountered in matmul.*",
)


PROCESSED_DATA_DIR = Path("../../data/processed")
MODELS_DIR = Path("../../models")
FEATURES_FILE = PROCESSED_DATA_DIR / "features.parquet"
METRICS_FILE = MODELS_DIR / "model_metrics.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
CV_FOLDS = 5
N_JOBS = 1
TIME_TRAIN_FRACTION = 0.6
TIME_VALID_FRACTION = 0.8
MIN_TIME_SPLIT_ROWS = 1000
ID_COLUMNS = ["person", "offer_id", "received_time"]
TARGET_COLUMN = "label"


def _build_time_split(
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Split by receipt chronology while keeping whole timestamps together."""
    unique_times = np.sort(features_df["received_time"].dropna().unique())
    if len(unique_times) < 3:
        raise ValueError("Not enough distinct receipt times for a chronological split.")

    train_idx = max(0, int(np.floor(len(unique_times) * TIME_TRAIN_FRACTION)) - 1)
    valid_idx = max(train_idx + 1, int(np.floor(len(unique_times) * TIME_VALID_FRACTION)) - 1)
    valid_idx = min(valid_idx, len(unique_times) - 2)
    if train_idx >= valid_idx:
        raise ValueError("Chronological split cutoffs collapsed onto same time block.")

    train_cutoff = unique_times[train_idx]
    valid_cutoff = unique_times[valid_idx]

    train_df = features_df.loc[features_df["received_time"] <= train_cutoff].copy()
    validation_df = features_df.loc[
        (features_df["received_time"] > train_cutoff)
        & (features_df["received_time"] <= valid_cutoff)
    ].copy()
    test_df = features_df.loc[features_df["received_time"] > valid_cutoff].copy()

    split_info = {
        "train_cutoff": float(train_cutoff),
        "validation_cutoff": float(valid_cutoff),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
    }
    return train_df, validation_df, test_df, split_info


def choose_split_strategy(features_df: pd.DataFrame) -> str:
    """Prefer a time split when each block is large enough and has both classes."""
    try:
        train_df, validation_df, test_df, _ = _build_time_split(features_df)
    except ValueError:
        return "random"

    splits = [train_df, validation_df, test_df]
    if any(len(split_df) < MIN_TIME_SPLIT_ROWS for split_df in splits):
        return "random"
    if any(split_df[TARGET_COLUMN].nunique() < 2 for split_df in splits):
        return "random"
    return "time"


def split_feature_matrix(
    features_df: pd.DataFrame,
    strategy: str = "auto",
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split features into train, validation, and test sets."""
    if strategy == "auto":
        strategy = choose_split_strategy(features_df)

    if strategy == "time":
        train_df, validation_df, test_df, _ = _build_time_split(features_df)
        return (
            train_df.reset_index(drop=True),
            validation_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    if strategy != "random":
        raise ValueError(f"Unsupported split strategy: {strategy}")

    train_val_df, test_df = train_test_split(
        features_df,
        test_size=TEST_SIZE,
        stratify=features_df[TARGET_COLUMN],
        random_state=random_state,
    )
    train_df, validation_df = train_test_split(
        train_val_df,
        test_size=VALIDATION_SIZE,
        stratify=train_val_df[TARGET_COLUMN],
        random_state=random_state,
    )
    return (
        train_df.sort_index().reset_index(drop=True),
        validation_df.sort_index().reset_index(drop=True),
        test_df.sort_index().reset_index(drop=True),
    )


def get_model_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return model-ready numeric feature columns."""
    return [
        column
        for column in features_df.columns
        if column not in ID_COLUMNS + [TARGET_COLUMN]
    ]


def prepare_xy(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature matrix into predictors and target."""
    feature_columns = get_model_feature_columns(features_df)
    return features_df[feature_columns].copy(), features_df[TARGET_COLUMN].astype(int)


def evaluate_binary_predictions(
    y_true: pd.Series, y_score: pd.Series, threshold: float = 0.5
) -> dict[str, float]:
    """Compute core binary classification metrics."""
    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def majority_class_baseline(y_train: pd.Series, y_eval: pd.Series) -> dict[str, float]:
    """Score majority-class baseline on an evaluation target."""
    majority_class = int(y_train.mode().iloc[0])
    scores = pd.Series([majority_class] * len(y_eval), index=y_eval.index, dtype=float)
    return evaluate_binary_predictions(y_eval, scores)


def rule_based_baseline(
    train_df: pd.DataFrame, eval_df: pd.DataFrame
) -> dict[str, float]:
    """Score simple income and offer-value rule baseline."""
    income_threshold = float(train_df["income_imputed"].median())
    rule_scores = (
        (eval_df["income_imputed"] > income_threshold)
        & (eval_df["reward"] > eval_df["difficulty"])
    ).astype(float)
    return evaluate_binary_predictions(eval_df[TARGET_COLUMN], rule_scores)


def build_model_registry() -> dict[str, object]:
    """Create baseline ML models used in project comparison."""
    models: dict[str, object] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        C=1.0,
                        random_state=RANDOM_STATE,
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=N_JOBS,
        )

    return models


def train_model(
    model: object, X_train: pd.DataFrame, y_train: pd.Series, cv: int = CV_FOLDS
) -> dict[str, float]:
    """Run stratified cross-validation for one model."""
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            module=r"sklearn\.utils\.extmath",
            category=RuntimeWarning,
        )
        scores = cross_validate(
            clone(model),
            X_train,
            y_train,
            cv=cv_splitter,
            scoring=["roc_auc", "f1", "precision", "recall"],
            n_jobs=N_JOBS,
        )

    summary = {}
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        summary[f"cv_{metric}_mean"] = float(scores[f"test_{metric}"].mean())
        summary[f"cv_{metric}_std"] = float(scores[f"test_{metric}"].std())
    return summary


def fit_and_evaluate_model(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    param_distributions: dict | None = None,
) -> tuple[object, dict[str, float]]:
    """Fit (and optionally tune) model on training split and score on validation."""
    if param_distributions:
        print(f"Running RandomizedSearchCV...")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=15,
            scoring='f1',
            cv=3,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbose=1
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                module=r"sklearn\.utils\.extmath",
                category=RuntimeWarning,
            )
            search.fit(X_train, y_train)
        
        fitted_model = search.best_estimator_
        print(f"Best parameters found: {search.best_params_}")
    else:
        fitted_model = clone(model)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                module=r"sklearn\.utils\.extmath",
                category=RuntimeWarning,
            )
            fitted_model.fit(X_train, y_train)

    validation_scores = fitted_model.predict_proba(X_validation)[:, 1]
    metrics = evaluate_binary_predictions(y_validation, pd.Series(validation_scores))
    return fitted_model, metrics


def save_model(model: object, path: Path) -> None:
    """Persist a fitted model with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def main() -> None:
    """Train baselines and production models from final feature matrix."""
    features_df = pd.read_parquet(FEATURES_FILE)
    split_strategy = choose_split_strategy(features_df)
    train_df, validation_df, test_df = split_feature_matrix(
        features_df, strategy=split_strategy
    )
    X_train, y_train = prepare_xy(train_df)
    X_validation, y_validation = prepare_xy(validation_df)

    metrics: dict[str, dict[str, float | int | str]] = {}
    metrics["split"] = {
        "strategy": split_strategy,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
    }
    if split_strategy == "time":
        _, _, _, split_info = _build_time_split(features_df)
        metrics["split"].update(split_info)
    metrics["environment"] = {
        "xgboost_available": XGBClassifier is not None,
    }
    metrics["majority_class"] = majority_class_baseline(y_train, y_validation)
    metrics["rule_based"] = rule_based_baseline(train_df, validation_df)

    # Define search spaces for ALL models
    param_dists = {
        "logistic_regression": {
            # Note the 'classifier__' prefix because LR is inside a Pipeline
            'classifier__C': uniform(0.1, 10.0), 
        },
        "random_forest": {
            'n_estimators': randint(100, 300),
            'max_depth': [None, 10, 20, 30],
            'min_samples_leaf': randint(1, 5)
        },
        "xgboost": {
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 8),
            'n_estimators': randint(100, 300),
            # scipy.stats uniform takes (loc, scale), so uniform(0.6, 0.4) searches 0.6 to 1.0
            'subsample': uniform(0.6, 0.4),      
            'colsample_bytree': uniform(0.6, 0.4) 
        }
    }

    for model_name, model in build_model_registry().items():
        print(f"\nTraining and tuning {model_name}...")
        cv_metrics = train_model(model, X_train, y_train)
        
        dist = param_dists.get(model_name)
        
        fitted_model, validation_metrics = fit_and_evaluate_model(
            model,
            X_train,
            y_train,
            X_validation,
            y_validation,
            param_distributions=dist
        )
        save_model(fitted_model, MODELS_DIR / f"{model_name}.joblib")
        metrics[model_name] = {**cv_metrics, **validation_metrics}

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
