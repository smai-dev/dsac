# Python Visualization Gallery: Seaborn and Plotly

## Goal

Demonstrate complementary techniques for displaying distributions, relationships, and changes over time in Plotly's built-in Gapminder dataset.

## Workflow

1. Load the example records and remove missing observations.
2. Compare numeric fields through pair and joint density views.
3. Inspect distributions with violin, boxen, swarm, and strip plots.
4. Examine correlations, faceted scatter plots, and a fitted relationship.
5. Produce continent-level summaries and time-series charts.
6. Render two interactive **2D** figures with Plotly; these do not use `scatter_3d`.

## Reading the figures

The different displays emphasize distinct aspects of the same records. Long-tailed GDP distributions may compress typical observations on a linear axis, and per-country faceting can create many small panels. The visual trend lines are exploratory—not proof of a causal link between national GDP and life expectancy.

## Environment notes

Use the Python environment defined in the repository-level setup guide. These notebooks use a built-in Plotly dataset, so its existing field names (`lifeExp`, `gdpPercap`, etc.) are kept unchanged to preserve the examples. On a Linux virtual machine, Plotly's **2D** HTML visualizations can generally render without the WebGL requirements of Plotly 3D scenes.

Reference PDFs from the supplied archive are retained byte-for-byte for optional reading.
