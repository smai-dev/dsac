# Binary Classification: Income Prediction

**Notebook:** `classification_adult.ipynb`  
**Goal:** Predict whether an individual's recorded income is `>50K` or `<=50K` using a dataset of 947 records containing numerical and categorical attributes.

## Approach

1. Explore the income labels and demographic and employment attributes; encode categorical features and standardise numerical features for logistic regression.
2. Compare **logistic regression**, **decision trees** and **XGBoost**. Use a held-out test set, cross-validation and hyperparameter search; control decision-tree complexity through pruning.
3. Evaluate accuracy alongside **precision, recall, F1-score, confusion matrices and ROC-AUC**.

## Findings

- The tuned logistic-regression model reached approximately **82.6% cross-validation accuracy** and **80% test accuracy**.
- The tuned XGBoost model reached approximately **84.1% cross-validation accuracy**, but **79% test accuracy**. Its reported test ROC-AUC was **0.90**, versus **0.88** for logistic regression. A higher cross-validation score did not translate into higher held-out accuracy in this experiment.

## Learning outcomes

Feature encoding and scaling; supervised classification; regularisation and pruning; grid/random hyperparameter search; validation versus test performance; and metric-driven interpretation of model trade-offs.

*Scope: an illustrative dataset exercise, not a validated model for real-world income decisions.*
