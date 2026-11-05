# ggplot2 Exploration: From Raw Records to Interpretable Visuals

## Objective

Use built-in R datasets (`diamonds`, `iris`, and `mpg`) to move from initial summaries to progressively richer statistical graphics. The notebook and companion R script examine sampling, transformations, overplotting, distribution shapes, and grouping variables without changing the source data.

## Analysis sequence

1. **Describe the data.** Inspect initial records, feature types, and numerical summaries before plotting.
2. **Compare measurements.** Use species-specific iris scatter plots and examine how diamond carat relates to price.
3. **Reduce crowding.** Contrast raw and logarithmic scales, take samples of the diamonds dataset, and vary transparency.
4. **Reveal distributions.** Compare histograms, density curves, and category counts; review how binning influences interpretation.
5. **Add context.** Use shape, color, size, facets, boxplots, violin plots, and fitted trends to show additional dimensions.

## Reading the results

The figures are exploratory, not evidence of causal relationships. Changes in axis scale, sampling, or smoothing can reveal different structures without changing the underlying observations. A carat-by-dimension product is used as a **volume proxy**, not a physical volume measurement verified from additional data.

## Local files

- `ggplot2_exploration.ipynb` — step-by-step R notebook (R kernel).
- `ggplot2_exploration.R` — companion script using the same examples.
- `data/` — reference CSV, `.data`, `.dat`, and Excel files retained from the supplied archive; the plotted `diamonds`/`iris`/`mpg` datasets come from R packages rather than these CSV files.

## Execution notes

Install the `ggplot2` package and choose an R notebook kernel. Some unsupervised sampling calls do not set a seed, so sampled points may differ between runs; the 500-row subset uses an explicit seed. Original notebook outputs have been retained for reference, but its figures should be regenerated locally.
