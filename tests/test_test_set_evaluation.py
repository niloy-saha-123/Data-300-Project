"""Tests for held-out test split evaluation."""

import json

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models import test_set_evaluation as tse
from src.models.train import RANDOM_STATE, prepare_xy, split_feature_matrix


def test_run_test_evaluation_writes_metrics(tmp_path, monkeypatch) -> None:
    """Synthetic feature matrix: metrics files written; baselines + model present."""
    rows = []
    for i in range(80):
        rows.append(
            {
                "person": f"p{i}",
                "offer_id": f"o{i % 3}",
                "received_time": float(i * 5),
                "label": i % 2,
                "income_imputed": 40000.0 + i * 500,
                "reward": 3.0,
                "difficulty": 2.0,
                "age_imputed": 30.0,
                "age_missing": 0,
                "income_missing": 0,
                "membership_duration_days": 100,
                "gender_F": 1,
                "gender_M": 0,
                "gender_O": 0,
                "gender_unknown": 0,
                "n_transactions_before": 1,
                "total_spend_before": 10.0,
                "avg_spend_before": 10.0,
                "days_since_last_transaction": 1.0,
                "days_since_last_transaction_missing": 0,
                "offers_received_before": 0,
                "offers_completed_before": 0,
                "offer_completion_rate_before": 0.0,
                "offer_type_bogo": 1,
                "offer_type_discount": 0,
                "duration_days": 7,
                "channel_web": 0,
                "channel_email": 1,
                "channel_mobile": 0,
                "channel_social": 0,
                "reward_to_difficulty_ratio": 1.5,
            }
        )
    features_df = pd.DataFrame(rows)

    train_df, _, test_df = split_feature_matrix(
        features_df, strategy="random", random_state=RANDOM_STATE
    )
    X_train, y_train = prepare_xy(train_df)
    model = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(tse, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(tse, "TEST_METRICS_JSON", reports_dir / "test_set_metrics.json")
    monkeypatch.setattr(tse, "TEST_METRICS_CSV", reports_dir / "test_set_model_metrics.csv")
    monkeypatch.setattr(
        tse,
        "load_available_models",
        lambda: {"logistic_regression": model},
    )

    payload = tse.run_test_evaluation(features_df=features_df)

    assert payload["split"]["test_rows"] == len(test_df)
    assert len(payload["models"]) == 1
    assert (reports_dir / "test_set_metrics.json").exists()
    assert (reports_dir / "test_set_model_metrics.csv").exists()
    saved = json.loads((reports_dir / "test_set_metrics.json").read_text(encoding="utf-8"))
    assert saved["split"]["strategy"] == "random"
    assert len(saved["baselines"]) == 2
