# Multivariate Data Visualization in R: Parallel Coordinates, Heatmaps, and Time-Series Plots

This document explains the content of both:

- a standalone **R script (`.R`)** containing the same analysis; and
- the corresponding **Jupyter notebook (`.ipynb`)** using an R kernel.

The code focuses on **multivariate data visualization**: displaying several variables simultaneously in order to reveal patterns, similarities, differences, clusters, and temporal trends.

The main datasets are:

- `iris`, the standard R dataset containing four numerical flower measurements and a species label;
- `appliances`, an external dataset loaded from `data/data_1/appliances.dat`.

The overall workflow is

$$
\text{Load libraries}
\rightarrow
\text{Inspect data}
\rightarrow
\text{Scale variables}
\rightarrow
\text{Visualize multiple dimensions}
\rightarrow
\text{Identify patterns}.
$$

---

## 1. R Script and Jupyter Notebook Versions

The same R commands can be used in either an `.R` script or an `.ipynb` notebook.

In an R script, commands are normally executed sequentially:

$$
C_1 \rightarrow C_2 \rightarrow \cdots \rightarrow C_n,
$$

where $C_i$ is the $i$-th command.

In a Jupyter notebook, the code is divided into cells:

$$
\text{execute cell}
\rightarrow
\text{inspect result}
\rightarrow
\text{continue analysis}.
$$

The notebook format is convenient for visualization because each table or figure appears directly after the corresponding code cell.

For reproducibility, notebook cells should still be executed in their logical order because later cells may depend on objects created earlier.

---

# 2. Loading the Required R Packages

```r
library(ggplot2)
library(GGally)
#library(MASS)
library(fmsb)
library(pheatmap)
library(dplyr)
```

Several libraries are loaded because different visualization techniques require different packages.

### `ggplot2`

`ggplot2` provides the grammar-of-graphics framework used for many R visualizations.

Conceptually,

$$
\text{plot}
=
\text{data}
+
\text{aesthetic mappings}
+
\text{geometries}
+
\text{scales}
+
\text{themes}.
$$

### `GGally`

`GGally` extends `ggplot2` with tools for multivariate visualization.

In this analysis it provides `ggparcoord()`, which creates **parallel-coordinate plots**.

### `MASS`

The line

```r
#library(MASS)
```

is commented out in the notebook.

The function `parcoord()` is commonly provided by the `MASS` package. If it is not already available in the R session, load the package explicitly:

```r
library(MASS)
```

### `fmsb`

`fmsb` contains functions useful for statistical and multivariate visualization.

### `pheatmap`

`pheatmap` is used to construct heatmaps with optional hierarchical clustering.

### `dplyr`

`dplyr` provides tools for data transformation and manipulation.

---

# 3. Inspecting the `iris` Dataset

The first step is to inspect the data.

```r
head(iris)
```

`head()` displays the first observations of the dataset.

The `iris` dataset contains four numerical measurements:

- `Sepal.Length`
- `Sepal.Width`
- `Petal.Length`
- `Petal.Width`

and one categorical variable:

- `Species`

Each flower can therefore be represented by a four-dimensional numerical vector

$$
\mathbf{x}_i
=
\left(
x_{i1},
x_{i2},
x_{i3},
x_{i4}
\right).
$$

The species variable identifies the class associated with each observation.

The three species are:

- `setosa`
- `versicolor`
- `virginica`

---

# 4. Summary Statistics

```r
summary(iris)
```

`summary()` provides descriptive statistics for each numerical variable.

For a numerical variable $X$, the output includes values such as:

- minimum;
- first quartile $Q_1$;
- median $Q_2$;
- mean;
- third quartile $Q_3$;
- maximum.

The arithmetic mean is

$$
\bar{x}
=
\frac{1}{n}
\sum_{i=1}^{n} x_i.
$$

The range is

$$
R
=
x_{\max} - x_{\min}.
$$

The interquartile range is

$$
IQR
=
Q_3-Q_1.
$$

These statistics provide an initial view of the location and spread of each variable before constructing multivariate plots.

---

# 5. Why Multivariate Visualization?

A conventional scatter plot normally displays two numerical dimensions:

$$
(x_i,y_i).
$$

The `iris` dataset contains four numerical features, so a single two-dimensional scatter plot cannot display the complete feature vector directly.

Multivariate visualization attempts to represent

$$
\mathbf{x}_i
=
(x_{i1},x_{i2},\ldots,x_{ip})
$$

for $p>2$.

The notebook explores this using:

- parallel-coordinate plots;
- heatmaps;
- hierarchical clustering;
- multiple line plots.

---

# 6. Parallel Coordinates with `GGally`

```r
ggparcoord(
  data = iris,
  columns = 1:4,
  groupColumn = 5,
  scale = "std",
  showPoints = TRUE
)
```

A **parallel-coordinate plot** represents several numerical variables simultaneously.

Instead of using perpendicular $x$- and $y$-axes, the variables are represented by parallel vertical axes.

For the `iris` dataset:

```text
Sepal.Length   Sepal.Width   Petal.Length   Petal.Width
     |              |              |              |
     | \            |            / |              |
     |  \___________|___________/  |______________|
     |              |              |              |
```

Each observation becomes a polyline crossing all axes.

For observation $i$,

$$
\mathbf{x}_i
=
(x_{i1},x_{i2},x_{i3},x_{i4})
$$

is represented by a line joining its value on each of the four axes.

---

## 7. Selecting the Variables

The argument

```r
columns = 1:4
```

selects the four numerical columns:

$$
X_1 = \text{Sepal.Length},
$$

$$
X_2 = \text{Sepal.Width},
$$

$$
X_3 = \text{Petal.Length},
$$

$$
X_4 = \text{Petal.Width}.
$$

The fifth variable, `Species`, is not treated as a numerical coordinate.

Instead, it is used to group observations.

---

# 8. Grouping by Species

```r
groupColumn = 5
```

The fifth column of `iris` is `Species`.

Therefore, the lines are grouped according to the class

$$
G_i
\in
\{
\text{setosa},
\text{versicolor},
\text{virginica}
\}.
$$

Grouping helps determine whether flowers belonging to the same species produce similar multivariate profiles.

If lines belonging to one species follow a distinct path through the coordinate axes, that suggests the numerical measurements contain information useful for separating that species from the others.

---

# 9. Standardization

The parallel-coordinate plot uses:

```r
scale = "std"
```

The variables in a multivariate dataset may have different centers and spreads.

Standardization places them on comparable scales.

For a variable $X_j$, a standard score is

$$
z_{ij}
=
\frac{x_{ij}-\bar{x}_j}{s_j},
$$

where:

- $x_{ij}$ is observation $i$ for variable $j$;
- $\bar{x}_j$ is the mean of variable $j$;
- $s_j$ is its standard deviation.

After standardization, a value of

$$
z=0
$$

corresponds approximately to the variable mean.

A positive value means the observation lies above the mean, while a negative value means it lies below the mean.

Standardization is important because otherwise a variable with a numerically larger scale could visually dominate the plot.

---

# 10. Showing Individual Points

```r
showPoints = TRUE
```

This displays the individual coordinate values in addition to the lines connecting them.

The plot therefore represents both:

- the value of each feature;
- the multivariate trajectory of each observation.

Parallel coordinates are particularly useful for detecting:

- groups with similar profiles;
- variables that help distinguish groups;
- unusual observations;
- correlations between neighboring dimensions.

---

# 11. Repeated Parallel-Coordinate Plot

The notebook contains the same command again:

```r
ggparcoord(
  data = iris,
  columns = 1:4,
  groupColumn = 5,
  scale = "std",
  showPoints = TRUE
)
```

This reproduces the same visualization.

In a notebook, repeating a plot can be useful when experimenting interactively or when preparing the plot before exporting it.

---

# 12. Saving a `ggparcoord` Plot to PNG

```r
png(
  "ggparcoord_parallel_plot.png",
  width = 1200,
  height = 800
)

ggparcoord(
  data = iris,
  columns = 1:4,
  groupColumn = 5,
  scale = "std",
  showPoints = TRUE
)

dev.off()
```

The `png()` function opens a PNG graphics device.

The dimensions

$$
1200 \times 800
$$

specify the image width and height in pixels.

The plot generated after `png()` is written to the file rather than only being displayed interactively.

Finally,

```r
dev.off()
```

closes the graphics device and completes the image file.

The workflow is therefore

$$
\texttt{png()}
\rightarrow
\text{draw plot}
\rightarrow
\texttt{dev.off()}.
$$

This is useful for saving plots for:

- reports;
- presentations;
- papers;
- websites;
- later analysis.

---

# 13. Parallel Coordinates with `parcoord()`

The notebook also creates a parallel-coordinate plot using another R function.

```r
par(
  mar = c(5, 5, 4, 2) + 0.1
)

par(
  cex = 1.2
)

parcoord(
  iris[, 1:4],
  col = as.factor(iris$Species),
  main = "Parallel Coordinates Plot of Iris Dataset",
  lwd = 2,
  pch = 19
)
```

This illustrates another implementation of the same multivariate visualization concept.

The numerical data supplied to `parcoord()` are

```r
iris[, 1:4]
```

so the visualization again operates on the four measurements.

---

# 14. Adjusting Plot Margins

```r
par(
  mar = c(5, 5, 4, 2) + 0.1
)
```

`par()` modifies graphical parameters.

The `mar` argument specifies plot margins approximately in the order

$$
(\text{bottom},\text{left},\text{top},\text{right}).
$$

Here the margins are based on

$$
(5,5,4,2)+0.1
=
(5.1,5.1,4.1,2.1).
$$

Increasing margins can prevent axis labels and titles from being clipped.

---

# 15. Adjusting Text Size

```r
par(
  cex = 1.2
)
```

The `cex` parameter controls the relative size of graphical text and symbols.

A value of

$$
\texttt{cex}=1.2
$$

means approximately $120\%$ of the default size.

---

# 16. Species as the Grouping Variable

The color argument is

```r
col = as.factor(iris$Species)
```

This converts the species variable to a factor and uses its categories to distinguish the lines.

Conceptually,

$$
\text{Species}
\longrightarrow
\text{line group/color}.
$$

The purpose is again to determine whether the four-dimensional profiles differ between species.

---

# 17. Line Width and Plot Title

```r
main = "Parallel Coordinates Plot of Iris Dataset"
```

adds a descriptive title.

```r
lwd = 2
```

sets the line width.

Increasing line width can improve visibility when the plot is displayed or exported.

The notebook also specifies

```r
pch = 19
```

as a plotting-symbol parameter.

---

# 18. Saving the `parcoord()` Plot

The parallel-coordinate plot is also exported to a PNG file.

```r
png(
  "parcoord_parallel_plot.png",
  width = 1200,
  height = 800
)

parcoord(
  iris[, 1:4],
  col = as.factor(iris$Species),
  main = "Parallel Coordinates Plot of Iris Dataset",
  lwd = 2,
  pch = 19
)

dev.off()
```

Again, the graphics-device sequence is

$$
\boxed{
\text{open device}
\rightarrow
\text{render visualization}
\rightarrow
\text{close device}
}
$$

with:

```r
png(...)
```

opening the device and

```r
dev.off()
```

closing it.

---

# 19. Scaling the `iris` Measurements

Before constructing a heatmap, the notebook standardizes the numerical variables explicitly.

```r
iris_scaled <- scale(
  iris[, 1:4]
)
```

For every variable $j$,

$$
z_{ij}
=
\frac{x_{ij}-\bar{x}_j}{s_j}.
$$

This gives each feature approximately:

$$
\text{mean}=0
$$

and

$$
\text{standard deviation}=1.
$$

This is particularly important in distance-based visual analysis.

Without scaling, a variable with a larger numerical range could contribute disproportionately to distances between observations.

---

# 20. Species Colors

The notebook defines a species-to-color mapping:

```r
annotation_colors <- list(
  Species = c(
    setosa = "red",
    versicolor = "green",
    virginica = "blue"
  )
)
```

This defines a categorical mapping:

$$
\begin{aligned}
\text{setosa} &\rightarrow \text{red},\\
\text{versicolor} &\rightarrow \text{green},\\
\text{virginica} &\rightarrow \text{blue}.
\end{aligned}
$$

Such annotations help connect clusters in the numerical data to known class labels.

---

# 21. Heatmap of the Scaled `iris` Data

The notebook then generates a heatmap:

```r
pheatmap(
  iris_scaled,
  cluster_rows = TRUE,
  cluster_cols = TRUE,
  annotation_col = data.frame(
    Species = iris$Species
  ),
  annotation_colors = annotation_colors
)
```

A heatmap converts numerical values into colors.

If

$$
Z =
[z_{ij}]
$$

is the standardized data matrix, then each matrix element $z_{ij}$ is represented by a color.

The basic mapping is

$$
z_{ij}
\longrightarrow
\text{color intensity}.
$$

This allows a large number of values to be examined simultaneously.

---

# 22. Structure of the Heatmap Matrix

The numerical `iris` matrix has the structure

$$
150 \text{ observations}
\times
4 \text{ variables}.
$$

Therefore,

$$
Z \in \mathbb{R}^{150\times4}.
$$

Rows correspond to flowers, while columns correspond to the four measurements.

A heatmap makes it possible to identify observations with similar high/low measurement patterns.

---

# 23. Hierarchical Clustering

The options

```r
cluster_rows = TRUE
cluster_cols = TRUE
```

request clustering of both observations and variables.

Hierarchical clustering begins with a notion of distance between vectors.

For two observations $\mathbf{x}_i$ and $\mathbf{x}_k$, the Euclidean distance is

$$
d(\mathbf{x}_i,\mathbf{x}_k)
=
\sqrt{
\sum_{j=1}^{p}
(x_{ij}-x_{kj})^2
}.
$$

Clustering then groups objects that are relatively close according to the chosen distance and linkage criterion.

The output can be represented as a **dendrogram**.

Conceptually,

```text
observations
     |
     +--- similar group
     |       |
     |       +--- observation
     |       +--- observation
     |
     +--- another group
             |
             +--- observation
             +--- observation
```

The heatmap reorders rows and columns according to these clustering results.

---

# 24. Interpreting the Heatmap

The heatmap helps answer questions such as:

- Which observations have similar feature profiles?
- Which measurements vary together?
- Do species correspond to visible clusters?
- Are some observations unusual?
- Which variables help distinguish groups?

After standardization,

$$
z_{ij}>0
$$

means a measurement is above its variable mean, while

$$
z_{ij}<0
$$

means it is below its variable mean.

The color patterns therefore describe relative rather than raw measurement magnitude.

---

# 25. Note on the Species Annotation

The matrix supplied to `pheatmap()` has:

- observations in rows;
- measurements in columns.

Since `Species` describes each flower observation, the species labels conceptually annotate **rows**, not measurement columns.

The notebook currently contains:

```r
annotation_col = data.frame(
  Species = iris$Species
)
```

For a row annotation, the intended form is generally:

```r
annotation_row = data.frame(
  Species = iris$Species
)
```

with suitable row names if required by the plotting setup.

The statistical idea remains the same:

$$
\text{observation}
\longrightarrow
\text{species annotation}.
$$

This distinction is important because a column annotation would instead describe the four measurement variables.

---

# 26. Loading the Appliances Dataset

The notebook next reads an external dataset.

```r
appliances <- read.table(
  "data/data_1/appliances.dat",
  header = TRUE,
  sep = ",",
  stringsAsFactors = FALSE
)
```

The file is interpreted as comma-separated tabular data.

The main arguments are:

```r
header = TRUE
```

which indicates that the first row contains column names;

```r
sep = ","
```

which specifies a comma as the field separator;

and

```r
stringsAsFactors = FALSE
```

which prevents character columns from being automatically converted into factors in R versions where that behavior may otherwise occur.

The result is stored as:

```r
appliances
```

---

# 27. Inspecting the Appliances Data

```r
head(appliances)
```

This displays the first observations of the external dataset.

The subsequent plotting code indicates that the variables of interest include:

- `YEAR`
- `DISH`
- `DISP`
- `FRIG`
- `WASH`

The first variable is used as the time axis, while the remaining variables are plotted as multiple series.

---

# 28. Multiple Time-Series or Trend Lines

The notebook plots several variables against year:

```r
matplot(
  appliances$YEAR,
  appliances[, 2:5],
  type = "l",
  col = 1:4,
  lty = 1,
  xlab = "Year",
  ylab = "Value",
  main = "Year vs DISH, DISP, FRIG, WASH"
)
```

`matplot()` is useful when several columns need to be plotted against the same horizontal variable.

If year is denoted by $t$, the plot represents several functions or sequences:

$$
y_1(t),\quad
y_2(t),\quad
y_3(t),\quad
y_4(t).
$$

The code therefore visualizes four variables over the same sequence of years.

---

# 29. Selecting the Four Series

The expression

```r
appliances[, 2:5]
```

selects columns $2$ through $5$.

If these correspond to `DISH`, `DISP`, `FRIG`, and `WASH`, then the plotted matrix is

$$
Y =
\begin{bmatrix}
y_{11} & y_{12} & y_{13} & y_{14}\\
y_{21} & y_{22} & y_{23} & y_{24}\\
\vdots & \vdots & \vdots & \vdots\\
y_{n1} & y_{n2} & y_{n3} & y_{n4}
\end{bmatrix}.
$$

For each year $t_i$, the graph displays four corresponding values.

---

# 30. Drawing Lines

```r
type = "l"
```

requests line plots.

The observations for each variable are connected in year order.

This representation is useful for examining:

- increases and decreases;
- long-term trends;
- relative growth;
- convergence or divergence between variables;
- periods of rapid change.

---

# 31. Line Colors and Types

```r
col = 1:4
```

assigns four different plotting colors.

```r
lty = 1
```

uses a solid line style.

The conceptual mapping is

$$
\text{variable}
\longrightarrow
\text{line appearance}.
$$

Distinguishing the lines visually makes it possible to compare multiple series in one coordinate system.

---

# 32. Axis Labels and Title

The horizontal axis is labeled:

```r
xlab = "Year"
```

and therefore represents time.

The vertical axis is labeled:

```r
ylab = "Value"
```

The title is:

```r
main = "Year vs DISH, DISP, FRIG, WASH"
```

This gives the viewer the essential context required to interpret the visualization.

---

# 33. Adding a Legend

```r
legend(
  "topright",
  legend = c(
    "DISH",
    "DISP",
    "FRIG",
    "WASH"
  ),
  col = 1:4,
  lty = 1
)
```

A legend connects each line appearance to its corresponding variable.

The mapping is conceptually

$$
\{\text{visual line}\}
\leftrightarrow
\{\text{variable name}\}.
$$

Without a legend, the viewer would see multiple lines but would not know which variable each line represents.

---

# 34. Comparing the Visualization Techniques

The notebook uses three main forms of multivariate visualization.

## Parallel coordinates

Parallel coordinates are most useful when every observation contains several numerical measurements:

$$
\mathbf{x}_i
=
(x_{i1},x_{i2},\ldots,x_{ip}).
$$

They emphasize the **profile of each observation across variables**.

## Heatmaps

Heatmaps represent a complete matrix:

$$
X \in \mathbb{R}^{n\times p}.
$$

They emphasize:

- high and low values;
- clusters of observations;
- clusters of variables;
- large-scale patterns.

## Multiple line plots

Multiple line plots represent several variables against a common ordered dimension such as time:

$$
y_j(t),
\qquad
j=1,\ldots,p.
$$

They emphasize:

- trends;
- temporal changes;
- comparisons between series.

---

# 35. Parallel Coordinates vs Heatmaps

Both techniques visualize high-dimensional observations, but they emphasize different structures.

A parallel-coordinate plot treats each observation as a line:

$$
\mathbf{x}_i
\rightarrow
\text{polyline}.
$$

A heatmap treats each value as a colored matrix cell:

$$
x_{ij}
\rightarrow
\text{color}.
$$

Parallel coordinates are particularly useful for following individual observation profiles.

Heatmaps are often more effective for identifying large groups or clusters when the dataset contains many observations.

---

# 36. Importance of Scaling

Scaling is a central concept in this notebook.

Suppose two variables have very different numerical ranges:

$$
X_1 \in [0,10]
$$

and

$$
X_2 \in [0,10000].
$$

Without scaling, variation in $X_2$ may dominate a distance calculation or graphical comparison.

Standardization transforms each feature to

$$
Z_j
=
\frac{X_j-\mu_j}{\sigma_j}.
$$

This gives the variables a comparable interpretation in terms of deviations from their own means.

For multivariate analysis, this is often essential.

---

# 37. Pattern Recognition in the `iris` Data

The multivariate plots are designed to reveal whether flower species exhibit distinct measurement patterns.

The general statistical question is:

$$
P(\text{features}\mid\text{species})
$$

or, from a classification perspective,

$$
P(\text{species}\mid\text{features}).
$$

The notebook does not build a classifier, but its visualizations help investigate whether classification may be feasible.

For example, if one species occupies a clearly distinct region of feature space, its measurements may provide strong discriminating information.

---

# 38. From Visualization to Analysis

The notebook can be viewed as an exploratory stage preceding formal statistical analysis or machine learning.

A typical workflow is

$$
\boxed{
\text{Data}
\rightarrow
\text{Inspection}
\rightarrow
\text{Scaling}
\rightarrow
\text{Multivariate visualization}
\rightarrow
\text{Pattern discovery}
\rightarrow
\text{Modeling}
}
$$

Visualization can suggest:

- possible clusters;
- discriminating features;
- correlations;
- outliers;
- time trends;
- variables requiring transformation.

These observations can then guide later statistical modeling.

---

# 39. Main R Functions Used

The notebook introduces several useful R functions.

### Dataset inspection

```r
head()
summary()
```

These functions provide an initial understanding of the data.

### Parallel coordinates

```r
ggparcoord()
parcoord()
```

These functions represent high-dimensional observations as connected profiles across parallel axes.

### Standardization

```r
scale()
```

This transforms numerical variables to comparable standardized scales.

### Heatmaps

```r
pheatmap()
```

This visualizes a numerical matrix and can perform hierarchical clustering.

### Reading data

```r
read.table()
```

This imports an external tabular dataset.

### Multiple series

```r
matplot()
```

This plots several numerical series against a common horizontal axis.

### Plot annotations

```r
legend()
```

This identifies the meaning of different visual series.

### Saving plots

```r
png()
dev.off()
```

These functions open and close a PNG graphics device.

---

# 40. Overall Workflow

The notebook follows this structure:

```text
Load visualization libraries
          |
          v
Inspect iris
          |
          v
Create parallel-coordinate plots
          |
          +------> GGally::ggparcoord()
          |
          +------> MASS::parcoord()
          |
          v
Save visualizations as PNG
          |
          v
Standardize iris measurements
          |
          v
Create clustered heatmap
          |
          v
Load appliances dataset
          |
          v
Inspect appliances
          |
          v
Plot several variables over year
```

Mathematically, the two principal data structures can be viewed as:

$$
X_{\text{iris}}
\in
\mathbb{R}^{n\times p}
$$

for multivariate observations, and

$$
Y(t)
=
\left[
y_1(t),
y_2(t),
\ldots,
y_p(t)
\right]
$$

for multiple variables observed along an ordered dimension such as time.

---

# 41. Key Takeaway

The central idea of the analysis is that data with several variables often require visualization techniques beyond ordinary two-dimensional scatter plots.

The notebook combines:

$$
\boxed{
\text{Parallel Coordinates}
+
\text{Standardization}
+
\text{Heatmaps}
+
\text{Clustering}
+
\text{Multiple Trend Lines}
}
$$

to explore different aspects of multivariate structure.

The `.R` script and `.ipynb` notebook represent the same analytical workflow in two different execution formats:

$$
\text{same R analysis}
\rightarrow
\begin{cases}
\text{sequential R script},\\
\text{interactive Jupyter notebook}.
\end{cases}
$$

The objective is not simply to produce figures, but to use visualization to discover relationships, groups, unusual observations, and trends that can guide subsequent statistical analysis.
