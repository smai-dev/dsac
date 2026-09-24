# Regression — Predicting Height from Body Measurements

## Project overview

Explore **507 observations** containing height and **11 numerical predictors**, including shoulder width, pelvic breadth, wrist and elbow diameters, age, and weight. The goal is to estimate height, compare regression methods, and understand how feature selection and regularization affect predictive performance.

## Short tutorial

1. **Explore the data.** Check completeness and plot each predictor against height. Several body measurements show positive relationships with height, while age shows little apparent relationship in this dataset.
2. **Establish a baseline.** Fit a single-feature linear regression using shoulder width (`biacromial`). Evaluate predictions with cross-validated **RMSE** (typical prediction-error magnitude) and **R²** (variation in height explained by the model).
3. **Add predictors.** Fit a multiple linear regression using all 11 predictors. Compare its cross-validated scores with the single-feature baseline and inspect true-versus-predicted values and residuals.
4. **Compare modelling approaches.** Evaluate **Ridge**, **ElasticNet**, and **Support Vector Regression (SVR)**. Tune ElasticNet's regularization settings and SVR's kernel, `C`, and `epsilon`, then compare cross-validated results.

## Key findings

| Model | Cross-validated RMSE | Cross-validated R² |
|---|---:|---:|
| Linear regression — shoulder width only | 6.36 | 0.24 |
| Linear regression — all predictors | 5.53 | 0.38 |
| Ridge regression — all predictors | 5.53 | 0.38 |
| Tuned ElasticNet — all predictors | 5.51 | 0.39 |
| Tuned SVR — all predictors | 5.54 | 0.38 |

**Takeaway:** Combining body measurements improved the linear baseline. ElasticNet gave a **small** improvement in the reported cross-validation results; the methods remained close in performance. With R² around **0.38–0.39**, substantial variation in height is still unexplained, so the results are an exploratory modelling exercise rather than evidence of deployment-ready accuracy.

## Learning outcomes

**Exploratory data analysis · Feature–target relationships · Simple vs multiple linear regression · Cross-validation · RMSE and R² · Residual analysis · Ridge/ElasticNet regularization · SVR hyperparameter tuning · Evidence-based model comparison**

**Tools:** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook.
