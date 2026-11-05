# Income Classification — Model Comparison and Evaluation

**Notebook:** `classification_income.ipynb`  
**Dataset:** `data/income_profiles.csv`  
**Task:** Binary classification of the recorded income categories (`<=50K` and `>50K`)  
**Dataset size:** 947 observations, with a 70/30 train/test split (662/285 records)

## Objective

This project investigates how well demographic, education, and employment-related attributes separate two recorded income bands. It compares a linear classifier, decision trees, and gradient-boosted trees while emphasizing the distinction between cross-validation performance and evaluation on held-out observations.

## Analysis workflow

1. **Explore and represent the records.** Inspect the target distribution and input types. Encode categorical variables and standardize numerical features where needed.
2. **Fit a logistic-regression baseline.** Evaluate the initial configuration, then search over regularization, solver, and optimization settings.
3. **Control decision-tree complexity.** Compare unconstrained trees with pre-pruning and cost-complexity pruning. Use a grid search to select a tree configuration.
4. **Train boosted trees.** Explore an XGBoost configuration and tune parameters using randomized search with macro-F1 as the search objective.
5. **Evaluate model quality.** Compare cross-validation summaries, held-out classification reports, confusion matrices, and ROC curves.

## Results from the executed notebook

The figures below were transcribed from the **saved notebook outputs**; they were not regenerated for this documentation update. Cross-validation accuracy is the mean printed by the notebook's evaluation helper. Test accuracy is shown as reported by scikit-learn, so rounded values are approximate.

| Selected model | Cross-validation accuracy | Held-out test accuracy | Test observations |
| --- | ---: | ---: | ---: |
| Tuned logistic regression | **82.62%** (±3.94%) | **79%** | 285 |
| Tuned decision-tree pipeline | **80.66%** (±4.11%) | **80%** | 285 |
| Tuned XGBoost | **83.39%** (±3.05%) | **82%** | 285 |

### Logistic regression

- The baseline five-fold cross-validation mean was **80.96%**, increasing to **82.62%** after the parameter search.
- The selected settings included `C=0.1`, `penalty='l2'`, and `solver='lbfgs'`.
- The held-out classification report recorded **79% accuracy**. Recall was **0.70** for the lower-income class and **0.88** for the higher-income class (under the notebook's encoded class ordering).

### Decision trees

- The unconstrained tree achieved **76.44%** mean cross-validation accuracy.
- Limiting tree depth or applying cost-complexity pruning improved the reported cross-validation results to approximately **80.66%** for the respective experiments.
- The grid search selected `ccp_alpha=0.003`, `criterion='gini'`, and `max_depth=4`, with a **0.8140** best cross-validated **macro-F1** score. This is **not** an accuracy measurement.
- The subsequently evaluated decision-tree pipeline obtained **80.66%** cross-validation accuracy and **80%** held-out test accuracy.

### XGBoost

- Randomized search evaluated boosted-tree configurations, using **macro-F1** as the optimization criterion.
- The resulting estimator achieved **83.39%** mean cross-validation accuracy and **82%** test accuracy.
- The test-set classification report gave an **F1-score of 0.82 for each encoded class** (rounded to two decimal places).

## Interpretation

Among the selected models, XGBoost has the highest reported cross-validation and held-out accuracy in this run. The relationship between class-wise recall and aggregate accuracy illustrates why confusion matrices and precision/recall scores remain valuable alongside the headline percentages.

These comparisons reflect the notebook's specific preprocessing and model-evaluation paths; they are descriptive rather than a statistically controlled comparison of identical feature pipelines. The ROC chart offers a further perspective on discrimination, but numerical AUC values are not stated here because they were not printed as standalone numbers in the saved notebook outputs.

## Reproducibility and limitations

- Results above come from a **previously executed notebook** supplied with this documentation update. No code cells were rerun or altered.
- Some results may vary if the environment, library versions, or stochastic training behavior differ.
- The CSV was renamed for presentation only; its underlying values were retained. **File renaming is not data anonymization.**
- This is a learning exercise, not a validated system for real-world income decisions.
