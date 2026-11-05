# Regression Study: Estimating Height from Anthropometric Measurements

## Project overview

This notebook investigates the relationship between recorded body measurements and height, using **507 observations** and **11 numerical predictors**. It compares ordinary least-squares regression, Ridge, ElasticNet, and Support Vector Regression (SVR), with exploratory visualizations and cross-validation-based model comparisons.

The objective is to assess how different feature sets and regression approaches perform on this sample. This is an **exploratory modelling exercise**, not a validated height-estimation system for individual decisions.

## Dataset and exploratory analysis

- **Source file:** `data/body_measurements.csv`
- **Sample size:** 507 rows and 12 columns.
- **Response variable:** `height`.
- **Candidate predictors:** `biacromial`, `pelvicBreadth`, `bitrochanteric`, `chestDepth`, `chestDiam`, `elbowDiam`, `wristDiam`, `kneeDiam`, `ankleDiam`, `age`, and `weight`.
- **Data quality:** The saved `smd.info()` output shows 507 non-null values for every column; no missing entries are reported. `age` is an integer-valued numerical variable.

The notebook begins with descriptive statistics and scatter plots of each predictor against height. Several body measurements show positive visual associations, while age appears less strongly related in these plots. These trends are descriptive and do not establish causality or predictive reliability.

A reproducible synthetic feature, `randomVariable`, is also introduced with seed `43` to illustrate an uninformative relationship. It is added to `smd_copy` for visualization only and **is not included in the regression experiments**.

## Modelling workflow

1. **Single-variable baselines:** Evaluate separate ordinary least-squares models using shoulder breadth (`biacromial`), elbow diameter (`elbowDiam`), and weight.
2. **Multivariable regression:** Fit an ordinary least-squares model using all 11 predictors, then inspect observed-versus-fitted values and residuals.
3. **Regression plane:** Fit a separate two-predictor model using `biacromial` and `pelvicBreadth` to display observations and the fitted plane in three dimensions.
4. **Nonlinear model comparison:** Search over SVR kernels and the `C` and `epsilon` hyperparameters.
5. **Regularized models:** Evaluate default ElasticNet, tune its `alpha` and `l1_ratio`, then compare Ridge models with different penalty strengths.

The reported comparison is based on the notebook's **five-fold cross-validation** procedure. Models used for some explanatory plots are also fitted on the complete dataset; those plots show **in-sample fits**, not independent test predictions.

## Recorded model results

The figures below are transcribed from the **saved outputs in the executed notebook**. No model was rerun to prepare this documentation.

| Model and predictors | CV RMSE | CV explained variance* |
|---|---:|---:|
| Linear regression — shoulder breadth | 6.36 | 0.24 |
| Linear regression — elbow diameter | 6.42 | 0.21 |
| Linear regression — weight | 6.79 | 0.21 |
| Linear regression — shoulder and pelvic breadth | 6.17 | 0.28 |
| Linear regression — all 11 predictors | 5.53 | 0.38 |
| ElasticNet — default parameters, all predictors | 5.65 | 0.37 |
| **ElasticNet — tuned, all predictors** | **5.51** | **0.39** |
| Ridge — all predictors | 5.53 | 0.38 |
| SVR — tuned, all predictors | 5.54 | 0.38 |

*The `ads.py` helper prints the second metric as “R-squared”, but its implemented scoring uses scikit-learn's `explained_variance`. The table uses the actual scoring metric. Its RMSE is calculated as the square root of the mean cross-validated mean squared error, rather than as a mean of separately computed fold RMSEs.*

### Selected hyperparameters

The saved grid-search outputs report:

| Estimator | Selected settings | CV RMSE | CV explained variance |
|---|---|---:|---:|
| SVR | `C=10`, `epsilon=1`, `kernel='poly'` | 5.54 | 0.38 |
| ElasticNet | `alpha=0.1`, `l1_ratio=0.1` | 5.51 | 0.39 |

The Ridge penalty-strength experiment examines `alpha` values from `1e-8` through `10`. The displayed results remain close to **RMSE 5.53** and **explained variance 0.38** at the precision printed in the notebook.

## Interpretation

Using multiple measurements improves the reported cross-validation performance compared with the single-feature baselines. The tuned ElasticNet has the lowest RMSE (**5.51**) in this comparison, although the difference from the full-predictor linear regression (**5.53**), Ridge (**5.53**), and tuned SVR (**5.54**) is small.

Explained variance around **0.38–0.39** for the strongest reported models indicates that a substantial portion of variation remains unexplained by these fitted approaches. The model rankings should therefore be interpreted cautiously rather than as proof of a practically meaningful advantage.

### Evaluation limitations

- Hyperparameter selection and subsequent model evaluation use the same available sample; the table does **not** describe performance on an independent final test set or a nested cross-validation assessment.
- Observed-versus-predicted plots and the residual diagnostics use estimators fitted on the full data and are descriptive, in-sample checks.
- Predictors have different measurement scales. Coefficient magnitudes are not directly comparable without accounting for these scales, and scaling can matter for regularized and kernel-based methods.
- The helper's printed “R-squared” label should not be confused with a separately calculated `r2_score`.
- The numerical results are specific to the saved data, model settings, and software environment; reproducibility across environments should be checked by running the notebook locally.

## Visualization notes for VS Code and virtual machines

The **uploaded, executed notebook still uses Plotly** for the 3D shoulder-breadth/pelvic-breadth regression plane. Its saved result is an interactive Plotly figure. Plotly's 3D visualization requires WebGL, which may be disabled or unavailable in VS Code's embedded notebook viewer inside a virtual machine.

This is a rendering limitation, **not evidence of a regression failure**. The 2D plots in the notebook use Matplotlib/Seaborn and do not depend on WebGL. To avoid the WebGL requirement for the regression plane, its Plotly visualization can be replaced with a Matplotlib `projection='3d'` scatter/surface plot; **that replacement is not present in this uploaded notebook**.

## Local execution

Keep the notebook, utilities, and dataset in the following layout:

```text
regression_height_estimation.ipynb
ads.py
data/
    body_measurements.csv
```

Open the notebook with the appropriate Python environment in Jupyter or VS Code. The notebook calls the renamed helper `ad.evaluate_regression_cv(...)` from the corresponding updated `ads.py` module; ensure this updated module is available on the Python import path.

**Documentation status:** This Markdown file has been updated against the saved outputs and source code of the uploaded, executed `regression_height_estimation(1).ipynb`. The notebook's code, parameters, dataset, and stored results were not modified or rerun as part of this documentation update.
