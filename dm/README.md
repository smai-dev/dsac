# Student Success Modeling 

> **Scope:** exploratory classification using 1,000 synthetic student records. The project demonstrates a transparent data-mining workflow; it is not a validated system for making decisions about individual learners.

## Overview

The guiding question is whether study habits, attendance and contextual indicators can distinguish students in two **exam-grade bands**. CRISP-DM provides the structure: problem framing, data exploration, preparation, modeling and interpretation.

The target is derived from the existing numeric `exam_result`: **D/F → `AtRisk`** (scores below 60), **A/B/C → `NotAtRisk`**. These target category values and all numeric cutoffs are unchanged. The observed score and derived grade are removed before the leakage-aware models are trained.

## Project layout

| File | Focus |
|---|---|
| `1_business_understanding.ipynb` | Problem framing, data-use boundaries and success measures |
| `2_data_understanding.ipynb` | Schema inspection, visual exploration, correlations and quality checks |
| `3_data_preprocessing.ipynb` | Leakage demonstration, imputation, discretization, feature construction and class balancing |
| `4_data_modeling.ipynb` | Tuned Random Forest and XGBoost, per-class metrics, confusion matrices and ROC curves |
| `utils.py` | Shared plotting, cross-validation and reporting functions with clearer public names |
| `data/student_habits_performance.csv` | **Only dataset included**, 1,000 rows × 16 renamed columns |
| `ATTRIBUTE_MAPPING.md` | Original-to-revised column names and definitions |
| `requirements.txt`, `SETUP_LINUX.md` | Installation and local execution notes |

The two numbered alternatives were consolidated: the more complete **preprocessing `_1`** and the more informative **modeling `_2`** were retained. The omitted alternatives are not duplicated in this package.

## Data and labels

The CSV records and values are exactly those supplied. Only field names changed; see `ATTRIBUTE_MAPPING.md`. The input contains student context, learning routines, daily activities and the recorded final exam score. The field `student_key` remains a record identifier and is excluded from the predictive feature matrix.

The model target has **280 `AtRisk`** and **720 `NotAtRisk`** examples (28%/72%). The source file uses 91 literal `None` entries for `parental_schooling_tier`, corresponding to the missing categorical values discussed in the notebooks.

## Analysis sequence

### 1 — Explore

Inspect distributions and subgroup summaries, Pearson and rank correlations, outlier indicators and logical consistency rules. In the supplied notebook output, daily study time has a strong positive Pearson association with the final score (approximately **0.83**); the reported Spearman association between the well-being rating and score is approximately **0.323**. These are correlations in synthetic data, not causal effects.

### 2 — Prepare

Construct the target from letter-grade bands, then examine a deliberately **leaky baseline** that uses outcome-derived fields. The first decision tree reaches 100% validation accuracy because `exam_result` and `grade` directly reveal the label. Removing those fields reduces the recorded baseline accuracy to approximately **84.10%**.

Subsequent cells compare missing-value imputation, tree-based discretization, activity aggregation, feature selection, and class-weight or resampling choices. 

### 3 — Compare ensembles

The retained modeling notebook trains and tunes **Random Forest** and **XGBoost** using an 80/20 stratified partition, a one-component media feature, class handling and embedded selection. 

The *existing saved outputs* from the original modeling run report:

| Held-out metric (200 synthetic records) | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | 0.88 | 0.91 |
| `AtRisk` precision | 0.76 | 0.82 |
| `AtRisk` recall | 0.84 | 0.84 |
| `AtRisk` F1 | 0.80 | 0.83 |
| ROC–AUC (rounded) | 0.94 | 0.95 |

Reported cross-validation mean accuracies (approximately **93.84%** and **94.79%**) describe the processed training-side analysis and **are not** the final held-out accuracies. These results are historical notebook outputs, not measurements from a rerun of this revised package.
