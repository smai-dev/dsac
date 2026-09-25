# Predicting Academic Risk with CRISP-DM

**An end-to-end machine learning case study | Supervised classification, feature engineering, model evaluation**

## Project at a glance

**Problem.** Explore whether student habits, lifestyle, and self-reported well-being can help identify students who may need academic support before final exam scores become available.

**Approach.** Follow the Cross-Industry Standard Process for Data Mining (**CRISP-DM**) from problem definition and exploratory data analysis through preprocessing, model development, and evaluation. Compare an interpretable decision-tree baseline with **Random Forest** and **XGBoost** classifiers.

**Outcome.** The case study produced a prototype risk classifier and an analysis of predictive features, class imbalance, data leakage, and evaluation trade-offs. On a held-out sample of **200 synthetic records**, the reported final XGBoost model achieved **0.91 accuracy** and **0.84 recall for the AtRisk class**. These are experimental results on synthetic data, *not* evidence of deployment or real-world early-warning performance.

**Technical stack.** Python, Jupyter Notebook, pandas, NumPy, seaborn, matplotlib, SciPy, scikit-learn, imbalanced-learn, feature-engine, and XGBoost.

---

## 1. Business understanding: turn a support goal into an ML task

The proposed application is an **early-warning decision-support tool**: flag students who might benefit from academic and well-being support while there is still time to intervene. The project sets an *intended* 6–8-week advance-warning goal; it does **not** establish that lead time experimentally.

The data-mining task is **binary classification**. An exam score is first mapped to an A–F grade. Grades **D and F** (scores below 60 under the project's grading rule) define `AtRisk`; grades A–C define `NotAtRisk`.

**Learning outcome:** Express an operational question as a target variable, an appropriate model task, and evaluation criteria. For an early-warning application, correctly detecting the minority `AtRisk` group matters alongside overall accuracy; incorrectly flagging a student also carries a cost.

## 2. Data understanding: investigate before modeling

The source is a **synthetic dataset of 1,000 student records** with academic, behavioral, lifestyle, and demographic attributes. Examples include study hours, attendance, sleep, social-media and Netflix use, exercise, mental-health rating, internet quality, and parental education. In the derived target, **280 records are AtRisk** and **720 are NotAtRisk**: a 28%/72% class split.

The exploratory workflow combined summary statistics, histograms, grouped comparisons, scatter plots, correlation matrices, and statistical tests:

- **Study hours and exam score:** a strong positive correlation of approximately **0.83** within this dataset.
- **Mental-health rating and exam score:** a positive rank association (reported Spearman's **ρ = 0.323**, *p* < 0.001).
- **Social-media and Netflix time:** relatively weak negative correlations with exam score (around **−0.17** for each measure).
- **Data quality:** **91 missing parental-education values**; outlying values were investigated rather than automatically discarded.

Pearson correlation was used for numerical relationships; **Spearman** and **Kendall** rank correlations for ordinal attributes; and group comparisons included **t-tests** and **one-way ANOVA**. Box plots, multivariate scatter plots, and **Isolation Forest** supported the outlier investigation; **PCA** was used to display high-dimensional observations in two dimensions.

**Learning outcome:** Choose analyses according to feature type, investigate unusual observations, and distinguish **association from causation**. Findings describe patterns in the synthetic sample; they do not demonstrate that changing a behavior will change a student's outcome.

## 3. Data preparation: construct a usable prediction problem

### 3.1 Find and remove target leakage

An initial decision-tree baseline reported **100% cross-validation accuracy**. The result was explained by the inclusion of `exam_score` and the derived `grade`: both directly determine the target label and would be unavailable when an early prediction is needed. Removing these proxy fields reduced reported baseline accuracy to **84.10%**, a more meaningful starting point for this task. The unique `student_id` was also removed.

**Learning outcome:** Ask, for every feature, *“Would this value genuinely be available at the moment a prediction is made?”* A perfect score can expose a design error rather than an exceptional model.

### 3.2 Prepare features and classes

The experimentation included:

1. **Stratified 80/20 train–test splitting** to preserve the `AtRisk` / `NotAtRisk` proportions.
2. **Mode imputation** for missing parental education, compared experimentally with KNN-based imputation.
3. **Categorical encoding** and **standardization** to prepare features for model training.
4. **PCA-based feature construction:** combine social-media and Netflix usage into a one-component `media_usage_hours_pca` feature while retaining the original variables.
5. **Decision-tree-based supervised discretization:** learn useful intervals for continuous attributes rather than relying only on fixed bins.
6. **BorderlineSMOTE:** synthesize minority-class training examples near difficult class boundaries; do **not** oversample the held-out test set.
7. **Embedded feature selection** using `SelectFromModel` and a tree-based estimator.

An XGBoost-based feature-selection pass retained **eight features**: `exercise_frequency`, `mental_health_rating`, `netflix_hours`, `part_time_job`, `sleep_hours`, `social_media_hours`, `study_hours_per_day`, and `media_usage_hours_pca`. This subset was used in the reported final ensemble comparison.

**Learning outcome:** Build a reproducible sequence of transformations, distinguish fitting a transformation from applying it, and understand why feature selection may retain variables that have weak **individual** correlations but contribute in a **multivariate** model.

## 4. Modeling: establish a baseline and compare ensembles

**Decision tree — interpretable baseline.** A decision tree repeatedly partitions observations using feature-based rules. Its depth, split size, leaf size, and pruning parameters control complexity. After preprocessing and randomized tuning, the report gives **0.88 held-out accuracy** and **0.84 AtRisk recall** for the tuned tree.

**Random Forest — bagging.** Multiple trees are trained using data resampling and feature randomness; their predictions are aggregated. This can reduce variance relative to an individual tree. The final experiment tuned tree count, depth, feature sampling, minimum split/leaf sizes, and bootstrap behavior.

**XGBoost — gradient boosting.** Trees are added sequentially to reduce the model's loss. Tuning included the number of trees, learning rate, depth, row/column subsampling, split controls, and L1/L2 regularization.

The notebooks used **RandomizedSearchCV with five-fold stratified cross-validation** to compare hyperparameter combinations, then assessed the selected models on a separate 200-record test subset. The reported final searches examined **300 Random Forest configurations** and **500 XGBoost configurations**.

**Learning outcome:** Explain the difference between a single tree, a bagged ensemble, and a boosted ensemble; control overfitting; and keep model selection distinct from final evaluation.

## 5. Evaluation: read the right metric for the right question

The final notebook and report provide the following **held-out test-set** results (200 records, including 56 AtRisk cases):

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Overall accuracy | 0.88 | 0.91 |
| AtRisk precision | 0.76 | 0.82 |
| AtRisk recall | 0.84 | 0.84 |
| AtRisk F1-score | 0.80 | 0.83 |
| ROC–AUC (reported) | 0.94 | 0.95 |

**How to interpret them.** `AtRisk` **recall** asks how many genuinely at-risk cases the classifier identifies; **precision** asks how many of its at-risk flags are correct. **F1** balances the two. **ROC–AUC** summarizes discrimination across thresholds, rather than performance at just one decision threshold. The confusion matrices are useful for examining false alarms and missed at-risk cases explicitly.

The source also reports **93.84% (Random Forest)** and **94.79% (XGBoost)** *cross-validation mean accuracy on the processed training data*. These are **not held-out test accuracies** and should not be presented to recruiters as such. The held-out figures above are the clearer summary of this experiment.

**Learning outcome:** Report class-specific performance, separate cross-validation from hold-out evaluation, and interpret apparently strong aggregate scores in the context of a minority group.

## 6. Practical findings and limits

**What the experiment demonstrated:** A full data-mining workflow can uncover informative patterns, expose a label-leakage problem, construct new features, handle class imbalance, and compare interpretable and ensemble classifiers on a defined prediction task. Study time and mental-health rating appeared prominently in the analyses, but a prediction alone does not explain an individual's circumstances or prescribe an intervention.

**What remains unproven:** The dataset is **synthetic** and contains **1,000 records**; the work did not validate predictions on an independent institution, demonstrate 6–8-week lead time, measure any actual improvement in student retention, or deploy an operational support system. Predictions should complement—not replace—human academic and well-being judgment.

**Methodological caveat for a production-grade follow-up:** The supplied final notebooks fit the PCA feature on the full dataset **before** the train/test split, and perform BorderlineSMOTE and feature selection **before** the model-selection cross-validation folds. Although the notebooks remove direct target proxies and keep the final test set out of SMOTE, these ordering choices can make validation scores optimistic. A stronger future evaluation would place every learned preprocessing step **inside each training fold** (using an appropriate pipeline), tune the entire pipeline with nested or otherwise properly isolated validation, retain an untouched test set, and validate on independent real-world data. This is an improvement opportunity, not a claim that the current experiment has already completed that validation.

## 7. Recruiter-facing learning outcomes

This case study provides concrete experience in:

- **End-to-end ML methodology:** CRISP-DM, problem framing, label definition, exploratory analysis, preprocessing, modeling, evaluation, and scope/limitations.
- **Data analysis and statistics:** descriptive statistics, visualizations, correlation by measurement scale, statistical tests, missing-data and outlier analysis.
- **Feature engineering:** categorical encoding, imputation, scaling, PCA, supervised discretization, and embedded selection.
- **Classification:** decision trees, Random Forest, XGBoost, regularization, class-imbalance techniques, and randomized hyperparameter search.
- **Evaluation discipline:** stratified splitting, cross-validation, hold-out testing, confusion matrices, precision, recall, F1, ROC–AUC, and recognition of leakage risks.
- **Responsible interpretation:** separating synthetic-data findings from field evidence, avoiding causal claims, and considering the human consequences of false positives and false negatives.

**Portfolio summary:** *End-to-end CRISP-DM case study developing and evaluating an academic-risk classifier on 1,000 synthetic student records. Investigated target leakage, engineered behavioral features, addressed class imbalance, and compared tuned decision-tree, Random Forest, and XGBoost models. The final XGBoost experiment reported 91% held-out accuracy and 84% AtRisk recall on a 200-record synthetic test subset, with explicit discussion of validation and generalization limits.*
