# Starbucks Rewards ML Project

End-to-end pipeline for predicting which customers respond to Starbucks offers, plus segmentation, uplift scoring, and budget simulation.

## What ships

- Raw JSON parsing to clean parquet tables
- Leakage-safe target construction: `received -> viewed -> completed` inside offer window
- Demographic, behavioral, and offer features
- Time-based train / validation / test split when chronology supports it
- Logistic Regression, Random Forest, and XGBoost
- Validation and held-out test reports
- Feature importance plots and SHAP summary
- K-Means customer segments with profile exports
- T-learner uplift model and budget allocation simulation

## Data

Download [Starbucks Rewards dataset](https://www.kaggle.com/datasets/blacktile/starbucks-app-customer-reward-program-data) and place these files in `data/raw/`:

- `portfolio.json`
- `profile.json`
- `transcript.json`

`data/raw/` stays local and should never be edited in place.

## Key modeling rules

- Positive label means customer received, viewed, and then completed offer before expiry.
- Completed-without-view stays negative.
- Informational offers are excluded from classification target.
- Behavioral features only use events that happened before each offer receipt.
- Prior-offer history uses observed events, not future labels from unresolved offers.

## Environment

Project expects conda env `Data300`.

```bash
conda activate Data300
pip install -r requirements.txt
```

## Run pipeline

Run from repo root:

```bash
python -m src.data.parse_raw
python -m src.data.build_target
python -m src.data.merge
python -m src.features.customer_features
python -m src.features.offer_features
python -m src.features.build_features
python -m src.models.train
python -m src.models.evaluate
python -m src.models.explain
python -m src.clustering.kmeans
python -m src.models.test_set_evaluation
python -m src.models.uplift
python -m src.models.budget_simulation
```

## Main outputs

Processed data:

- `data/processed/features.parquet`
- `data/processed/offer_response.parquet`
- `data/processed/customer_segments.parquet`
- `data/processed/uplift_features.parquet`

Models:

- `models/logistic_regression.joblib`
- `models/random_forest.joblib`
- `models/xgboost.joblib`
- `models/uplift_treatment_model.joblib`
- `models/uplift_control_model.joblib`

Reports:

- `reports/model_comparison_table.csv`
- `reports/validation_model_metrics.csv`
- `reports/test_set_metrics.json`
- `reports/cluster_profiles.csv`
- `reports/uplift_metrics.json`
- `reports/budget_simulation.json`

Figures:

- ROC, PR, and confusion matrices
- Logistic / RF / XGBoost importance plots
- SHAP summary
- Elbow, silhouette, cluster heatmaps
- Uplift Qini and cumulative gain charts
- Budget allocation comparison

## Notebooks

Notebooks stay lightweight and read from `src/` outputs:

- `01_data_exploration.ipynb`
- `02_eda.ipynb`
- `03_feature_engineering.ipynb`
- `04_modeling.ipynb`
- `05_clustering.ipynb`
- `06_uplift.ipynb`

## Tests

```bash
pytest tests
```
