# Presentation Outline

## 1. Problem

- Predict real offer response
- Avoid wasting offers on customers who buy anyway

## 2. Data

- portfolio, profile, transcript
- 61,042 actionable customer-offer rows after target build

## 3. Leakage-Safe Target

- received -> viewed -> completed
- completed-without-view stays negative

## 4. Feature Engineering

- demographics
- pre-offer spend behavior
- offer design and channel features

## 5. Model Comparison

- show test-set table
- highlight XGBoost best ROC-AUC

## 6. Explainability

- top XGBoost drivers
- logistic coefficients for directionality
- SHAP summary

## 7. Customer Segments

- 4 clusters
- completion rate by segment and offer type

## 8. Uplift Modeling

- informational offers as control
- Qini and cumulative gain

## 9. Budget Simulation

- greedy uplift ranking
- compare against random baseline

## 10. Takeaways

- strongest model
- strongest customer segments
- business use for targeting
