# Descriptive Statistics and Distribution Diagnostics with R

## Objective

Use an external census sample and generated observations to examine location, spread, shape, and the assumptions behind normality checks. The notebook reports numerical summaries and plots; it does not train a predictive model.

## Workflow

1. **Load and inspect the census data.** Read `data/census_respondents.csv`, verify the schema, and summarize the available records.
2. **Characterize age.** Plot a histogram and boxplot; compare the mean, 10% trimmed mean, median, standard deviation, and mode.
3. **Quantify distribution shape.** Calculate skewness and kurtosis, noting that a single number cannot capture every departure from a reference distribution.
4. **Simulate contrasting distributions.** Generate binomial, lognormal, and Student-t samples as provided in the notebook.
5. **Use diagnostic tests and plots.** Run Shapiro–Wilk and Anderson–Darling tests and review a normal Q–Q plot. A statistical test is a measure of evidence against a distributional hypothesis, not a universal verdict on data quality.

## Interpretation and limitations

Normality tests are sensitive to sample size; the large Student-t sample in particular may detect small departures. Random draws for the lognormal and Student-t examples have no explicit seed in the original workflow, so exact values can differ on rerun. The source code uses a *descriptive* census sample; it does not justify statements about a larger population.

## Compatibility change

The original code loaded its data into `US_census` but attempted `head(us_census)`, which is a different symbol in case-sensitive R. The revised notebook calls `head(US_census)`. The `Age` column is renamed `respondent_age` consistently across the CSV and code; the numeric age values and statistical formulas remain unchanged.

## Dependencies

R, `IRkernel`, `moments`, and `nortest`. Other core descriptive calculations use base R.
