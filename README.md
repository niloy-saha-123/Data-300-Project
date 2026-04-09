# Data 300 Project
### Predicting Customer Response to Promotional Offers - Starbucks Rewards Program

**Niloy Saha & Pranav Azad**

## Status
The repository is currently scaffolded for implementation. The first build phase sets up the package layout, portable dependencies, and the modeling rules that must stay fixed throughout the project.

## Project Overview
This project builds a machine learning pipeline on the Starbucks Rewards dataset to predict whether a customer will complete a promotional offer. The intended workflow is:

1. Parse and flatten the raw JSON files.
2. Build a clean `(customer, offer)` response table.
3. Engineer demographic, behavioral, and offer-level features.
4. Train and compare Logistic Regression, Random Forest, and XGBoost.
5. Interpret the best model with feature importance and SHAP.
6. Segment customers with K-Means clustering.
7. Optionally add uplift modeling and a simple budget allocation simulation.

## Dataset
The project uses three JSON files from the [Starbucks Rewards Kaggle dataset](https://www.kaggle.com/datasets/blacktile/starbucks-app-customer-reward-program-data):

| File | Description |
| --- | --- |
| `portfolio.json` | Offer metadata: type, reward, duration, channels |
| `profile.json` | Customer demographics and membership date |
| `transcript.json` | Time-ordered customer events and transactions |

Download the dataset from Kaggle here:
[https://www.kaggle.com/datasets/blacktile/starbucks-app-customer-reward-program-data](https://www.kaggle.com/datasets/blacktile/starbucks-app-customer-reward-program-data)

After downloading, place `portfolio.json`, `profile.json`, and `transcript.json` in `data/raw/` before running the pipeline. Raw data stays local and is not committed to Git.

## Fixed Modeling Rules
These decisions are part of the project spec and should not drift between branches.

- Positive label: a customer-offer pair is labeled `1` only when the customer receives the offer, views it, and then completes it within the offer duration window.
- Completed without view: rows where the offer is completed without a prior view are labeled `0`, because they reflect organic behavior rather than offer response.
- Informational offers: excluded from the classification target because they cannot be completed in the same way as BOGO and discount offers.
- Leakage prevention: all behavioral features must be computed using only events that occurred before the current offer's `received_time`.

## Repository Layout
```
data/
  raw/            # Input JSON files, kept local and never modified
  processed/      # Parsed and engineered outputs
notebooks/        # EDA and analysis notebooks
src/
  data/           # Parsing, target construction, dataset merges
  features/       # Demographic, behavioral, and offer features
  models/         # Training, evaluation, and model explanation
  clustering/     # Customer segmentation
  utils/          # Shared helpers
reports/
  figures/        # Exported report figures
models/           # Serialized trained models, kept local
tests/            # Unit tests for pipeline logic
```

## Setup
Python `3.10+` is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the optional uplift modeling phase, install the stretch dependency separately:

```bash
pip install -r requirements-uplift.txt
```

## Planned Pipeline
The implementation is planned around these scripts and outputs:

```bash
python src/data/parse_raw.py
python src/data/build_target.py
python src/data/merge.py
python src/features/customer_features.py
python src/features/offer_features.py
python src/models/train.py
python src/models/evaluate.py
```

## Contributing
Use feature branches named `niloy/<feature-name>` and `pranav/<feature-name>`. Open pull requests into `main`; do not commit project work directly to `main`.
