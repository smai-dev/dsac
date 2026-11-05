# Multivariate Patterns in R

## Purpose

Explore observations across several dimensions using parallel-coordinate plots, standardized heatmaps, and grouped time-series lines. These views help identify feature relationships and clusters that are difficult to see in isolated scatter plots.

## Workflow

1. **Inspect the iris measurements.** Use the built-in dataset and review each species before plotting.
2. **Compare standardized attributes.** Draw parallel-coordinate charts with `GGally::ggparcoord()` and `MASS::parcoord()`. Scaling prevents a variable with larger numerical units from visually dominating the plot.
3. **Inspect row patterns.** Standardize the four iris measurements and visualize them with `pheatmap`, labeling **observation rows** by species.
4. **Examine annual series.** Read `data/core_reference/appliance_trends.dat` and compare the four recorded appliance measures using common year values.
5. **Export selected figures.** Save the parallel-coordinate variants as PNG files without altering the underlying datasets.

## Interpretation

Parallel coordinates reveal record-level profiles, heatmaps show relative similarities in scaled features, and line charts emphasize changes over time. The heatmap's hierarchical clustering is exploratory; it does not establish natural categories or causal relationships.

## Compatibility adjustments

The R notebook commented out `MASS` even though it calls `parcoord()`. This revision explicitly loads that package. Its heatmap annotation is attached to **rows** (`annotation_row`), matching the 150 iris records; the `annotation_col` targeted the four feature columns instead. These are narrow execution/annotation corrections, not changes to the observations or distance calculations.

## Data and references

The CSV and appliance headers were made more descriptive. Every data row remains unchanged. Two compressed reference datasets and the sample retail spreadsheets are included unchanged under `data/`. Existing PNGs are reference exports and will be replaced only if you execute the plotting cells.
