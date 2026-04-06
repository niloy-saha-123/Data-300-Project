# Data 300 Project
### Predicting Customer Response to Promotional Offers — Starbucks Rewards Program

**Niloy Saha & Pranav Azad**

---

## Overview
This project applies machine learning to the Starbucks Rewards Mobile App dataset to predict whether a customer will complete a promotional offer based on their demographics and past behavior. The pipeline covers data parsing, feature engineering, classification modeling, customer segmentation, and optional uplift analysis.

---

## Dataset

Three JSON files from the [Starbucks Rewards Kaggle dataset](https://www.kaggle.com/datasets/blacktile/starbucks-app-customer-reward-program-data):

| File | Description |
|---|---|
| `portfolio.json` | Details on 10 promotional offers (type, reward, duration, channels) |
| `profile.json` | Demographics for 17,000 simulated customers |
| `transcript.json` | 300,000+ event logs (offer received, viewed, completed, transactions) |

Place all three files in `data/raw/` before running anything.

---

## Setup

**Python 3.10+ recommended.**

```bash
git clone https://github.com/your-username/data-300-project.git
cd data-300-project
pip install -r requirements.txt
```

---

## How to Run

Run each step in order. Each script reads from `data/processed/` and writes back to it.

```bash
# 1. Parse and flatten the raw JSON files
python src/data/parse_raw.py

# 2. Build the (customer, offer, label) response table
python src/data/build_target.py

# 3. Merge demographics and offer metadata
python src/data/merge.py

# 4. Build the final feature matrix
python src/features/customer_features.py
python src/features/offer_features.py

# 5. Train and evaluate models
python src/models/train.py
python src/models/evaluate.py
```

Notebooks in `notebooks/` can be run independently after the processed data files exist.

---

## Project Structure

```
data-300-project/
├── data/
│   ├── raw/                  # Original JSON files — do not modify
│   └── processed/            # Cleaned and engineered DataFrames (Parquet/CSV)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_clustering.ipynb
│   └── 06_uplift.ipynb
├── src/
│   ├── data/                 # Parsing, target construction, merging
│   ├── features/             # Feature engineering functions
│   ├── models/               # Training, evaluation, SHAP analysis
│   ├── clustering/           # K-Means segmentation
│   └── utils/                # Shared I/O and plotting helpers
├── models/                   # Saved model files (joblib) — gitignored
├── reports/
│   └── figures/              # Exported figures for the report
├── tests/                    # Unit tests for data pipeline
├── requirements.txt
└── README.md
```

---

## Models

Three classifiers are trained and compared:

- Logistic Regression (baseline)
- Random Forest
- XGBoost

Evaluation metrics: ROC-AUC, F1-score, Precision-Recall AUC, and confusion matrix. Models are evaluated with 5-fold stratified cross-validation and a held-out test set.

---

## Key Design Decisions

**Target variable:** A customer-offer pair is labeled positive (1) only if the customer viewed the offer and then completed it within the offer's duration window. Completions without a prior view are labeled negative — those represent organic purchases, not offer-driven behavior.

**Leakage prevention:** All behavioral features (transaction counts, spend history) are computed using only events that occurred before the offer was received.

**Informational offers excluded:** Since informational offers have no completion mechanic, they are excluded from the classification task.

---

## Requirements

Key dependencies:

```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
joblib
econml        # optional, for uplift modeling
black
flake8
```

Full list with pinned versions in `requirements.txt`.

---

## Contributing

Branch naming: `name/short-description` (e.g., `niloy/parse-json`, `pranav/eda-notebook`).
Open a pull request to merge into `main`. Do not commit directly to `main`.
Large files (models, raw data) are gitignored — store them locally or via shared Drive.