# Binary Classification: Student Outcomes

**Notebook:** `classification_student.ipynb`  
**Goal:** Explore prediction of first-year student **Pass/Fail** outcomes from 1,131 records containing academic scores, study habits and learning-related attributes.

## Approach

1. Inspect the data, consider missing values, encode the target label and explore feature standardisation.
2. Train and compare **decision trees**, **k-nearest neighbours (k-NN)** and **support vector machines (SVM)**. Explore SVM kernel/regularisation settings and use cross-validation.
3. Examine confusion matrices, class-specific precision/recall/F1 and ROC curves rather than relying on overall accuracy alone.

## Findings

- The notebook's decision-tree discussion reports around **75% test accuracy**, with weaker classification of **Fail** than **Pass**.
- Its baseline SVM comparison highlights a failure mode: predictions favour **Pass** while missing **Fail** cases. In the notebook's comparison, **k-NN produces more balanced class-level results**.
- The accompanying report also examines a *separately tuned* SVM and reports improved results; its model settings should not be confused with the notebook's baseline comparison.

## Learning outcomes

Classification of numerical data; label encoding and feature scaling; distance-based versus margin-based learning; kernel selection and hyperparameter tuning; class imbalance; and the importance of per-class evaluation.

*Scope: an educational model-comparison exercise, not a validated system for student assessment.*
