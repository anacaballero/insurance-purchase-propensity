# Insurance Purchase Propensity Model

Binary classification model predicting vehicle insurance purchase likelihood for a portfolio of ~65,000 clients at a Swedish insurance company.

## Problem

The dataset presents severe class imbalance: only **12% of clients purchase** vehicle insurance (Target=1), while 88% do not. A naive classifier achieves 88% accuracy by always predicting non-purchase — but captures 0% of actual buyers. The goal is to maximize F1 score while preserving business utility across both classes.

**Dataset:** 65,499 clients, 14 features (anonymized)  
**Task:** Binary classification — `Target` ∈ {0: non-purchase, 1: purchase}

## Approach

### Exploratory Data Analysis
- Distribution analysis by segment: age, premium amount, customer tenure, gender, vehicle age/damage, sales channel, region
- High-conversion segments identified: Region 28 (20% conversion vs 12% average), channels 26 and 124 (34% and 33% of all purchases)
- Feature selection: `Körkort` dropped (99.8% uniform); `HarMotorförsäkring` retained as high-signal predictor

### Data Preparation
- One-Hot encoding for categorical variables (`Kön`, `HarMotorförsäkring`, `FordonsÅlder`, `FordonsSkada`, `kanal`, `region`)
- StandardScaler normalization for numerical features (`Ålder`, `Premie`, `TidSomKund`)
- 80/20 stratified train/test split

### Class Imbalance Strategies
1. **Class weighting** — differentially penalizes minority class misclassification
2. **Undersampling** — balanced training set (6,457 purchasers / 6,457 non-purchasers)
3. **Probability threshold tuning** — adjusts decision boundary post-training based on business objective

### Model Comparison

| Model | Strategy | F1 | Precision (purchase) | Recall (purchase) |
|-------|----------|----|----------------------|-------------------|
| Logistic Regression baseline | Class weights 0.4/0.6 | 0.08 | — | 0.04 |
| LR + GridSearchCV (5-fold) | Weights 0.22/0.78, C=20, L2 | **0.43** | **0.75** | 0.31 |
| LR + Undersampling + GridSearchCV | Balanced set, 7-fold | 0.38 | 0.26 | **0.98** |
| XGBoost | Threshold=0.22 | **0.44** | 0.62 | 0.34 |
| XGBoost + Undersampling | Threshold=0.21 | 0.40 | 0.30 | 0.72 |

### Key Finding

Model selection depends on the business objective:
- **Quality of leads (precision):** LR + GridSearchCV — 75% of flagged clients are true purchasers
- **Capture all buyers (recall):** LR + Undersampling — catches 98% of purchasers at the cost of precision
- **Best overall balance:** XGBoost without undersampling (F1=0.44, strongest AUC)

## Stack

`Python` `scikit-learn` `XGBoost` `pandas` `NumPy` `matplotlib` `seaborn`

## Data

Anonymized client records (not included in repo). Schema:

```
Kön, Ålder, Körkort, HarMotorförsäkring, FordonsÅlder,
FordonsSkada, Premie, TidSomKund, kanal, region, Target
```
