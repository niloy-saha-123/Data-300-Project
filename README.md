# Starbucks Rewards ML Project

Predict customer response to Starbucks promotional offers.

## Goal
- Build clean pipeline from raw JSON to predictions
- Predict if customer will complete offer
- Compare Logistic Regression, Random Forest, XGBoost
- Add EDA, feature engineering, clustering
- Optional: uplift modeling, budget simulation

## Dataset
Use [Starbucks Rewards Kaggle dataset](https://www.kaggle.com/datasets/blacktile/starbucks-app-customer-reward-program-data).

Put these files in `data/raw/`:
- `portfolio.json`
- `profile.json`
- `transcript.json`

Raw data stays local. Do not commit raw files.

## Fixed Rules
- Positive label = `received -> viewed -> completed` within offer window
- `completed without view = 0`
- Exclude informational offers from classification target
- Behavioral features must use only events before `received_time`

## Current Status
Done:
- repo setup
- raw parsing
- target build
- dataset merge
- demographic features
- behavioral features
- offer features
- final feature matrix
- training pipeline
- validation evaluation (ROC / PR / confusion) + CSV reports
- SHAP / feature importance (tree models) via `explain`
- clustering (K-Means, figures, segment tables)
- **held-out test metrics** via `test_set_evaluation` (after train)
- initial exploration + EDA + feature notebooks
- plotting config
- pytest suite (`pytest tests` works from repo root)

Not done:
- uplift modeling
- budget simulation
- optional notebooks `04_modeling` / `05_clustering` / `06_uplift`
- final report / slides

## Repo Layout
```text
data/
  raw/         local input files
  processed/   generated outputs
notebooks/     exploration and analysis
src/
  data/        parsing, target, merge
  features/    demographic, behavioral, offer, final matrix
  models/      training, evaluation, explainability
  clustering/  segmentation
  utils/       shared helpers
reports/
  figures/     exported figures
models/        saved trained models
tests/         unit tests
```

## Setup
Use conda env:

```bash
conda activate Data300
pip install -r requirements.txt
```

Optional uplift deps:

```bash
pip install -r requirements-uplift.txt
```

## Run Order
Run from the **repository root** (paths are relative to cwd).

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
```

`test_set_evaluation` scores the **held-out test split** using models saved under `models/`. Run it **once** after training when you want final numbers for the report. Outputs: `reports/test_set_metrics.json` and `reports/test_set_model_metrics.csv`.

Run tests:

```bash
pytest tests
```

## Team Split
- `niloy/*` and `pranav/*` feature branches
- no direct project work on `main`
- notebooks for exploration only
- reusable logic goes in `src/`
