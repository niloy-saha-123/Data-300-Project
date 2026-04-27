# Starbucks Rewards Offer Response Modeling

## Problem

Goal: predict which customer-offer pairs lead to a real offer response, not an organic purchase.

Positive label means:

1. offer received
2. offer viewed
3. offer completed before offer expiry

Completed-without-view stays negative. Informational offers stay out of classification target.

## Data Pipeline

Raw inputs come from `portfolio.json`, `profile.json`, and `transcript.json`.

Pipeline outputs:

- parsed tables in `data/processed/`
- target table in `offer_response.parquet`
- final modeling matrix in `features.parquet`

Behavioral features only use events before each offer receipt. Prior-offer history comes from observed transcript events, not future labels from unresolved offers.

## Features

Three feature blocks feed models:

- demographics: age, income, gender, membership tenure
- behavior: transaction count, spend history, recency, prior offer engagement
- offer metadata: type, difficulty, reward, duration, channel flags

## Models

Chronological split:

- train: receipts through hour 336
- validation: hour 337 to 408
- test: hour 409 onward

Compared models:

- Logistic Regression
- Random Forest
- XGBoost

Baselines:

- majority class
- simple income/reward rule

## Held-Out Test Results

| Model | ROC-AUC | AP | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7458 | 0.5382 | 0.5761 | 0.5145 | 0.6546 |
| Random Forest | 0.7665 | 0.5585 | 0.5465 | 0.5643 | 0.5298 |
| XGBoost | 0.7854 | 0.5860 | 0.5732 | 0.5810 | 0.5655 |

XGBoost is best overall on ROC-AUC and average precision. Logistic Regression keeps strongest recall and stays useful as most interpretable baseline.

## Explainability

Saved explainability outputs:

- logistic coefficient plot
- random forest importance plot
- XGBoost importance plot
- SHAP summary for best tree model

Feature importance files live under `reports/feature_importance/`.

## Clustering

K-Means selection picked `K = 4`.

Segment pattern:

- Cluster 1 and 3 show strongest response rates
- Cluster 2 is large and low-response
- Discount offers outperform BOGO for weaker segments

Cluster tables and figures live in `reports/cluster_profiles.csv`, `reports/cluster_offer_response.csv`, and `reports/figures/`.

## Uplift And Budgeting

Uplift setup uses informational offers as control and BOGO/discount offers as treatment. Outcome is any transaction inside offer window.

Key uplift outputs:

- treated-model ROC-AUC: `0.7887`
- control-model ROC-AUC: `0.7609`
- Qini AUC: `53.49`
- top 30% incremental gain: `242`

Budget simulation ranks actionable offers by uplift per reward dollar.

- budget: `13627.75`
- greedy expected incremental purchases: `124.21`
- random baseline mean: `51.17`

## Reproducibility

- conda env: `Data300`
- all random seeds fixed at `42`
- scripts run top-to-bottom from repo root
- `pytest tests` passes
- notebooks `01` through `06` execute with `jupyter nbconvert --execute`
