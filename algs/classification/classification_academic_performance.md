# Academic Performance Classification — Learning-Outcome Prediction

**Notebook:** `classification_academic_performance.ipynb`  
**Dataset:** `data/academic_profiles.csv`  
**Task:** Predict first-year student outcomes (`Pass` or `Fail`)  
**Dataset size:** 1,131 records

## Objective

This study evaluates how decision trees, support vector machines (SVMs), and k-nearest neighbours (k-NN) classify student outcomes from academic and learning-related indicators. The main question is not only which model has better aggregate accuracy, but also how reliably each method recognizes both outcome classes.

## Experimental workflow

1. **Explore and transform the data.** Review the available predictors, encode the target, and investigate feature standardization.
2. **Establish a decision-tree baseline.** Train a tree on a stratified 70/30 partition and evaluate its held-out predictions.
3. **Inspect an RBF-kernel SVM.** Review class-specific errors, its ROC-AUC, and alternative parameter settings.
4. **Search for an SVM configuration.** Compare kernel and regularization choices with five-fold grid search.
5. **Compare SVM with k-NN.** Evaluate both on a separate training partition using five-fold cross-validation and out-of-fold predictions.

## Results from the executed notebook

The results below are drawn from the **saved cell outputs**. They describe different experiments and should not be treated as one shared test set or a single uniform evaluation protocol.

### Decision-tree evaluation

| Evaluation | Accuracy | Records assessed |
| --- | ---: | ---: |
| Initial stratified held-out test | **75%** | 340 |
| Five-fold out-of-fold evaluation | **69%** | 1,131 |
| Ten-fold out-of-fold evaluation | **71%** | 1,131 |

The initial held-out confusion matrix was:

| Actual / Predicted | Fail | Pass |
| --- | ---: | ---: |
| **Fail** | 83 | 43 |
| **Pass** | 42 | 172 |

The corresponding class-wise report gave **precision, recall and F1 of 0.66** for `Fail`, and **0.80** for `Pass`. The test set contained 126 `Fail` and 214 `Pass` cases.

### SVM evaluation and selection

- The baseline RBF-kernel SVM obtained **63% held-out accuracy** on 340 examples, with **ROC-AUC = 0.6886** as printed by the notebook.
- Its predictions contained **no `Fail` classifications**: the confusion matrix was `[[0, 126], [0, 214]]`. Accordingly, `Fail` recall and F1 were both **0.00**. The apparent `Pass` F1 of approximately **0.77** does not compensate for this failure mode.
- A randomized parameter search printed `Best F1 Score: nan`; that result **does not establish a valid improvement in F1**.
- A separate five-fold grid search selected a **linear SVM with `C=1`**. The grid-search result is a model-selection output and should not be confused with the earlier RBF baseline's test score.

### k-NN versus linear SVM

The final comparison uses **791 training records**, standardized predictors, five-fold cross-validation, and out-of-fold predictions. Its reported values are **training-partition cross-validation estimates, not held-out test accuracy**.

| Metric | k-NN (`k=5`) | Linear SVM (`C=1`) |
| --- | ---: | ---: |
| Mean five-fold accuracy | **77.62%** | **67.76%** |
| `Fail` recall | **0.65** | **0.40** |
| `Pass` recall | **0.85** | **0.84** |
| `Fail` F1-score | **0.68** | **0.48** |
| `Pass` F1-score | **0.83** | **0.77** |

The k-NN model performed better on both aggregate cross-validation accuracy and recognition of `Fail` cases in this final comparison. The SVM identified many `Pass` cases but had substantially lower recall for `Fail`.

## Interpretation and reporting notes

Class-specific evaluation is essential here: a model that mostly predicts `Pass` can appear reasonable by overall accuracy while failing to identify students in the other outcome category. The evaluated k-NN configuration offers a more balanced error profile than the evaluated SVM configuration.

Two details matter when reading the notebook's visual outputs:

- The final k-NN/SVM ROC curves are drawn using predictions scored on the **training partition** after fitting, not on the separate held-out test set. They should not be presented as independent test-set ROC performance.
- In the ten-fold decision-tree cell, the printed classification report uses fresh ten-fold predictions, but the adjacent confusion-matrix visualization uses the matrix retained from the earlier five-fold cell. The written summary therefore relies on the **printed ten-fold report**, not that plot.

## Reproducibility and limitations

- The numerical figures above were taken from the user-provided executed notebook; **no cells were run and no model code was changed** for this Markdown update.
- The saved outputs may depend on library versions, data preparation and random partitions used in each experiment.
- The dataset's filename changed, but its records and values were retained. **A renamed CSV is not anonymized data.**
- This is an educational classification comparison, not a validated system for assessing individual students or determining educational interventions.
