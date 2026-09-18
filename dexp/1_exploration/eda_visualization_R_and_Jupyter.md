# Exploratory Data Analysis and Data Visualization with `ggplot2`

This document explains the content of both:

- the standalone **R script** containing the exploratory analysis and visualizations;
- the corresponding **Jupyter notebook (`.ipynb`)** containing the same R workflow in notebook cells.

Both versions perform essentially the same exploratory data analysis. The main difference is the execution environment:

- in an `.R` script, the code is normally executed sequentially as one R program;
- in a Jupyter notebook, the same R code is divided into cells that can be executed interactively and independently.

The analysis itself is therefore documented once in this file and applies to **both the R script and the notebook**.

This tutorial introduces **exploratory data analysis (EDA)** and common data-visualization techniques in R using `ggplot2`.

The examples use three datasets:

- `diamonds`: diamond characteristics such as carat, cut, color, clarity, dimensions, and price.
- `iris`: measurements of iris flowers from three species.
- `mpg`: fuel-economy information for different vehicles.

The general purpose of exploratory data analysis is to understand the structure of a dataset, identify relationships between variables, detect unusual observations, and formulate hypotheses before applying more formal statistical models.

A typical exploratory workflow is

$$
\text{Inspect data}
\rightarrow
\text{Visualize variables}
\rightarrow
\text{Identify patterns}
\rightarrow
\text{Investigate anomalies}
\rightarrow
\text{Model relationships}.
$$

---

## R Script and Jupyter Notebook Versions

The same commands can be used in both environments.

### R script

A standalone R file may contain commands sequentially:

```r
head(diamonds)
summary(diamonds)
str(diamonds)

ggplot(
  diamonds,
  aes(
    x = carat,
    y = price
  )
) +
  geom_point()
```

When the script is sourced or executed from top to bottom, each statement runs in sequence.

Conceptually,

$$
C_1 \rightarrow C_2 \rightarrow C_3 \rightarrow \cdots \rightarrow C_n,
$$

where $C_i$ denotes the $i$-th R command.

### Jupyter notebook

In the `.ipynb` version, the same R code is organized into separate code cells.

A notebook therefore supports an interactive workflow such as

$$
\text{run cell}
\rightarrow
\text{inspect output}
\rightarrow
\text{modify code}
\rightarrow
\text{run next cell}.
$$

This is particularly convenient for exploratory data analysis because tables, summaries, and plots can be examined immediately after each step.

### Important execution difference

In a script, execution order normally follows the physical order of the file.

In a notebook, cells may technically be executed in a different order. For reproducible analysis, they should still be run from top to bottom because later cells may depend on objects created earlier, for example:

```r
set.seed(639245)

dsmall <- diamonds[
  sample(nrow(diamonds), 500),
]
```

Later plots use `dsmall`, so the cell that creates it must be executed first.

The logical dependency is

$$
\texttt{diamonds}
\rightarrow
\texttt{dsmall}
\rightarrow
\text{plots using }\texttt{dsmall}.
$$

### Shared analytical content

Whether the code is stored in `.R` or `.ipynb`, the workflow covers the same topics:

- dataset inspection;
- descriptive statistics;
- scatter plots;
- logarithmic transformations;
- random sampling;
- reproducibility with `set.seed()`;
- aesthetic mappings;
- transparency and overplotting;
- smoothing and regression;
- bar charts;
- histograms;
- density plots;
- boxplots;
- jitter plots;
- faceting;
- heatmaps;
- violin plots;
- multivariate visualization.

The rest of this document therefore explains the analysis independently of the file format.

---

## 1. Inspecting a Dataset

Before plotting data, it is useful to inspect its structure and obtain summary statistics.

### First observations

```r
head(diamonds)
```

`head()` displays the first few rows of the dataset. It gives a quick view of the variables and their values.

For the `diamonds` dataset, important variables include:

- `carat`: diamond weight.
- `cut`: cut quality.
- `color`: diamond color category.
- `clarity`: clarity category.
- `depth`: total depth percentage.
- `table`: width of the top facet relative to the widest point.
- `price`: price in US dollars.
- `x`, `y`, `z`: physical dimensions of the diamond.

### Summary statistics

```r
summary(diamonds)
```

`summary()` gives descriptive statistics for each variable.

For a numerical variable $X$, typical values include:

- minimum,
- first quartile $Q_1$,
- median $Q_2$,
- mean,
- third quartile $Q_3$,
- maximum.

The mean of $n$ observations is

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i.
$$

The median is the central observation after sorting the data and is less sensitive to extreme values than the mean.

### Inspecting another dataset

```r
head(iris)

summary(iris)
```

The `iris` dataset contains measurements of sepals and petals for three iris species.

Its main numerical variables are:

- `Sepal.Length`
- `Sepal.Width`
- `Petal.Length`
- `Petal.Width`

and its categorical variable is:

- `Species`

### Dataset structure

```r
str(diamonds)
```

`str()` reports the internal structure of the dataset, including variable names, types, and example values.

This is useful for determining whether a variable is:

- numeric,
- integer,
- character,
- factor,
- ordered factor.

---

## 2. Scatter Plots

A scatter plot is useful for investigating the relationship between two numerical variables.

For observations

$$
(x_1,y_1), (x_2,y_2), \ldots, (x_n,y_n),
$$

a scatter plot places each observation as a point in a two-dimensional coordinate system.

### Petal length and petal width

```r
ggplot(
  iris,
  aes(
    x = Petal.Length,
    y = Petal.Width,
    color = Species
  )
) +
  geom_point()
```

The horizontal axis represents petal length and the vertical axis represents petal width.

Mapping `Species` to `color` adds a third variable to the visualization.

This allows us to inspect whether the relationship between petal length and width differs between species.

### Sepal length and sepal width

```r
ggplot(
  iris,
  aes(
    x = Sepal.Length,
    y = Sepal.Width,
    color = Species
  )
) +
  geom_point()
```

Again, color is used to distinguish the species.

Scatter plots can reveal:

- positive relationships,
- negative relationships,
- clusters,
- nonlinear patterns,
- outliers.

---

## 3. Diamond Carat and Price

A central relationship in the `diamonds` dataset is the relationship between diamond weight and price.

```r
ggplot(
  diamonds,
  aes(
    x = carat,
    y = price
  )
) +
  geom_point()
```

Each point represents one diamond.

In general, price increases as carat increases, but the relationship is not perfectly linear.

A simple linear relationship would have the form

$$
P = \beta_0 + \beta_1 C + \varepsilon,
$$

where:

- $P$ is price,
- $C$ is carat,
- $\beta_0$ is the intercept,
- $\beta_1$ is the slope,
- $\varepsilon$ represents unexplained variation.

The raw data suggest that the relationship is more complicated than a simple straight line.

---

## 4. Logarithmic Transformations

When two variables have a strongly skewed or multiplicative relationship, logarithmic transformations can make the pattern easier to study.

```r
ggplot(
  diamonds,
  aes(
    x = log(carat),
    y = log(price)
  )
) +
  geom_point()
```

The transformation is

$$
C' = \log(C)
$$

and

$$
P' = \log(P).
$$

A power-law relationship of the form

$$
P = aC^b
$$

becomes linear after taking logarithms:

$$
\log(P)
=
\log(a)
+
b\log(C).
$$

Therefore, a log-log plot is particularly useful when the relationship between two variables may approximately follow a power law.

---

## 5. Sampling Large Datasets

The `diamonds` dataset contains many observations.

```r
nrow(diamonds)
```

When a dataset is large, plotting every observation can be computationally expensive and visually crowded.

A random sample can provide a quicker exploratory view.

```r
ggplot(
  diamonds[sample(nrow(diamonds), 10000), ],
  aes(
    x = carat,
    y = price
  )
) +
  geom_point()
```

The same idea can be applied after a logarithmic transformation.

```r
ggplot(
  diamonds[sample(nrow(diamonds), 10000), ],
  aes(
    x = log(carat),
    y = log(price)
  )
) +
  geom_point()
```

If a dataset contains $N$ observations and we select $n$ observations, the sampling fraction is

$$
f = \frac{n}{N}.
$$

Random sampling can preserve the broad structure of the data while reducing plotting cost.

---

## 6. Diamond Volume and Outliers

The variables `x`, `y`, and `z` describe the dimensions of a diamond.

A simple approximate volume measure is

$$
V = xyz.
$$

We can plot this approximate volume against carat.

```r
ggplot(
  diamonds,
  aes(
    x = carat,
    y = x * y * z
  )
) +
  geom_point()
```

This plot may reveal visible outliers.

An outlier is an observation that differs substantially from the general pattern of the data.

For example, unusually large or zero values for physical dimensions may indicate:

- measurement errors,
- recording errors,
- unusual objects,
- genuine rare cases.

EDA helps identify such observations before further analysis.

---

## 7. Creating a Reproducible Random Sample

To make random sampling reproducible, we fix the random-number-generator seed.

```r
set.seed(639245)

dsmall <- diamonds[
  sample(nrow(diamonds), 500),
]
```

The seed ensures that repeated executions generate the same sample.

```r
summary(dsmall)
```

The sample contains $500$ observations.

```r
nrow(dsmall)
```

We can inspect approximate diamond volume using the sample.

```r
ggplot(
  dsmall,
  aes(
    x = carat,
    y = x * y * z
  )
) +
  geom_point()
```

Sampling allows us to experiment rapidly while retaining a representative subset of the data.

---

## 8. Mapping Variables to Visual Properties

One of the main ideas of `ggplot2` is the mapping of variables to aesthetics.

Common aesthetics include:

- horizontal position: `x`
- vertical position: `y`
- color: `color`
- shape: `shape`
- size: `size`
- transparency: `alpha`
- fill: `fill`

Conceptually,

$$
\text{data variable}
\longrightarrow
\text{visual property}.
$$

### Mapping color

```r
ggplot(
  dsmall,
  aes(
    x = carat,
    y = price,
    color = color
  )
) +
  geom_point()
```

The variable `color` is categorical, so each category is assigned a different visual color.

### Mapping shape

```r
ggplot(
  dsmall,
  aes(
    x = carat,
    y = price,
    shape = cut
  )
) +
  geom_point()
```

The variable `cut` is represented using different point shapes.

An alternative is to explicitly treat `cut` as an unordered factor:

```r
ggplot(
  dsmall,
  aes(
    x = carat,
    y = price,
    shape = factor(cut, ordered = FALSE)
  )
) +
  geom_point()
```

This emphasizes that the variable is being used as a categorical grouping variable for the plot.

---

## 9. Overplotting and Transparency

Large scatter plots often suffer from **overplotting**.

Overplotting occurs when many observations occupy similar positions, causing points to cover one another.

Transparency can reveal areas of high point density.

```r
ggplot(
  diamonds,
  aes(
    x = carat,
    y = price
  )
) +
  geom_point(alpha = 1/10)
```

Here,

$$
\alpha = \frac{1}{10} = 0.1.
$$

A lower alpha value makes each point more transparent.

```r
ggplot(
  diamonds,
  aes(
    x = carat,
    y = price
  )
) +
  geom_point(alpha = 1/100)
```

Now,

$$
\alpha = \frac{1}{100} = 0.01.
$$

An even more transparent version is:

```r
ggplot(
  diamonds,
  aes(
    x = carat,
    y = price
  )
) +
  geom_point(alpha = 1/200)
```

with

$$
\alpha = \frac{1}{200} = 0.005.
$$

Regions containing many overlapping observations appear darker than regions containing few observations.

---

## 10. Lines and Smooth Curves

A line can be added to data using `geom_line()`.

```r
ggplot(
  dsmall,
  aes(
    x = carat,
    y = price
  )
) +
  geom_line() +
  geom_smooth()
```

However, `geom_line()` connects observations according to their ordering and may not always be meaningful for independent observations such as diamonds.

A scatter plot is usually more appropriate.

```r
ggplot(
  dsmall,
  aes(
    x = carat,
    y = price
  )
) +
  geom_point() +
  geom_smooth()
```

`geom_smooth()` estimates the underlying relationship between the variables.

Conceptually, we assume

$$
Y = f(X) + \varepsilon,
$$

where $f(X)$ is an unknown smooth function.

### Controlling smoothness

```r
ggplot(
  dsmall,
  aes(
    x = carat,
    y = price
  )
) +
  geom_point() +
  geom_smooth(span = 0.1)
```

The `span` parameter controls the degree of smoothing.

A smaller span follows local fluctuations more closely.

For example:

```r
ggplot(
  dsmall,
  aes(
    x = carat,
    y = price
  )
) +
  geom_point(alpha = 0.4) +
  geom_smooth(span = 0.3)
```

Compare this with the default smoother:

```r
ggplot(
  dsmall,
  aes(
    x = carat,
    y = price
  )
) +
  geom_point(alpha = 0.4) +
  geom_smooth()
```

The choice of smoothness represents a trade-off between:

$$
\text{local detail}
\quad\text{and}\quad
\text{general trend}.
$$

---

## 11. Bar Charts

Bar charts are useful for categorical variables.

```r
ggplot(
  diamonds,
  aes(x = cut)
) +
  geom_bar()
```

`geom_bar()` automatically counts the number of observations in each category.

If $n_j$ is the number of observations belonging to category $j$, then the height of its bar is

$$
h_j = n_j.
$$

The same chart can be created for the smaller sample.

```r
ggplot(
  dsmall,
  aes(x = cut)
) +
  geom_bar()
```

---

## 12. Building a Plot Incrementally

One useful feature of `ggplot2` is that plots can be constructed layer by layer.

### Initial bar chart

```r
my_plot =
  ggplot(
    diamonds,
    aes(x = cut)
  ) +
  geom_bar()

print(my_plot)
```

### Adding count labels

```r
my_plot =
  my_plot +
  geom_text(
    stat = "count",
    aes(label = ..count..),
    vjust = -1
  )

print(my_plot)
```

The count of each category is now displayed above its bar.

### Increasing the text size

```r
my_plot =
  my_plot +
  theme(
    text = element_text(size = 14)
  )

print(my_plot)
```

### Modifying the vertical axis

```r
my_plot =
  my_plot +
  ylim(0, 35000)

print(my_plot)
```

The displayed range of the vertical axis becomes

$$
0 \leq y \leq 35000.
$$

### Changing the bar color

```r
my_plot =
  my_plot +
  geom_bar(
    fill = "lightsteelblue"
  )

print(my_plot)
```

### Flipping the coordinates

```r
my_plot =
  my_plot +
  coord_flip()

print(my_plot)
```

The categorical axis and count axis are exchanged, producing horizontal bars.

This illustrates the layered grammar used by `ggplot2`:

$$
\text{plot}
=
\text{data}
+
\text{aesthetics}
+
\text{geometries}
+
\text{statistics}
+
\text{scales}
+
\text{coordinates}
+
\text{theme}.
$$

---

## 13. Histograms

A histogram summarizes the distribution of a continuous variable.

The base R version is:

```r
hist(diamonds$price)
```

A `ggplot2` histogram is:

```r
ggplot(
  diamonds,
  aes(x = price)
) +
  geom_histogram(binwidth = 500)
```

If a bin has boundaries $a_j$ and $a_{j+1}$, then its count is

$$
n_j
=
\#\{x_i : a_j \leq x_i < a_{j+1}\}.
$$

The `binwidth` determines the width of each interval.

Here,

$$
\Delta = 500.
$$

Instead of specifying the width, we can specify the number of bins.

```r
ggplot(
  diamonds,
  aes(x = price)
) +
  geom_histogram(bins = 10)
```

Different bin choices may reveal or hide features of the distribution.

---

## 14. Density Plots

A density plot estimates the probability density of a continuous variable.

```r
ggplot(
  diamonds,
  aes(x = price)
) +
  geom_density()
```

A filled version is:

```r
ggplot(
  diamonds,
  aes(x = price)
) +
  geom_density(
    fill = "lightsteelblue4"
  )
```

A probability density $f(x)$ satisfies

$$
f(x) \geq 0
$$

and

$$
\int_{-\infty}^{+\infty} f(x)\,dx = 1.
$$

A common kernel density estimator has the form

$$
\hat{f}_h(x)
=
\frac{1}{nh}
\sum_{i=1}^{n}
K\left(
\frac{x-x_i}{h}
\right),
$$

where:

- $K$ is the kernel,
- $h$ is the bandwidth,
- $n$ is the sample size.

Like the `span` parameter in smoothing, the bandwidth controls how smooth the estimated density becomes.

---

## 15. Adding a Second Categorical Dimension

A bar chart can represent more than one categorical variable.

```r
my_plot =
  ggplot(
    diamonds,
    aes(
      x = cut,
      fill = clarity
    )
  ) +
  geom_bar()

my_plot
```

The main categories are given by `cut`, while the bars are subdivided using `clarity`.

This allows the joint distribution of two categorical variables to be inspected.

---

## 16. Color and Cut

Another categorical comparison is:

```r
my_plot =
  ggplot(
    diamonds,
    aes(
      x = color,
      fill = cut
    )
  ) +
  geom_bar()

my_plot
```

For each diamond color category, the bar displays the composition of cut categories.

This helps investigate whether the relative frequency of cut quality changes with color.

---

## 17. Histograms by Group

A continuous distribution can also be compared across categories.

```r
my_plot =
  ggplot(
    diamonds,
    aes(
      x = price,
      fill = cut
    )
  ) +
  geom_histogram(
    binwidth = 500
  )

my_plot
```

The variable `price` is continuous, while `cut` defines the groups.

The resulting visualization compares the price distributions of the cut categories.

---

## 18. Density Curves by Group

Density curves can be colored according to a categorical variable.

```r
my_plot =
  ggplot(
    diamonds,
    aes(
      x = price,
      color = cut
    )
  ) +
  geom_density()

my_plot
```

This is useful for comparing the **shape**, **location**, and **spread** of several continuous distributions.

---

## 19. Boxplots

A boxplot summarizes a numerical distribution using quartiles.

```r
my_plot =
  ggplot(
    diamonds,
    aes(
      x = cut,
      y = price
    )
  ) +
  geom_boxplot() +
  coord_flip() +
  ggtitle("Boxplots of Price by Cut")

my_plot
```

A standard boxplot is based on:

- first quartile $Q_1$,
- median $Q_2$,
- third quartile $Q_3$.

The interquartile range is

$$
IQR = Q_3 - Q_1.
$$

A common rule identifies potential outliers below

$$
Q_1 - 1.5\,IQR
$$

or above

$$
Q_3 + 1.5\,IQR.
$$

Boxplots provide a compact way to compare distributions across groups.

---

## 20. Jitter Plots

The following visualization compares diamond color with price per carat.

```r
my_plot =
  ggplot(
    diamonds,
    aes(
      x = color,
      y = price / carat
    )
  ) +
  geom_jitter()

my_plot
```

The derived variable is

$$
\text{price per carat}
=
\frac{\text{price}}{\text{carat}}.
$$

For categorical horizontal axes, many points would otherwise appear at exactly the same horizontal coordinate.

`geom_jitter()` adds small random displacements to improve visibility.

Conceptually,

$$
x_i' = x_i + \epsilon_i,
$$

where $\epsilon_i$ is a small random perturbation used only for visualization.

---

## 21. Linear Regression with the `mpg` Dataset

The `mpg` dataset can be used to explore the relationship between engine displacement and highway fuel economy.

```r
ggplot(
  mpg,
  aes(
    displ,
    hwy,
    color = factor(cyl)
  )
) +
  geom_point() +
  stat_smooth(method = "lm")
```

Here:

- `displ` is engine displacement,
- `hwy` is highway fuel economy,
- `cyl` is the number of cylinders.

The fitted linear model is approximately

$$
Y = \beta_0 + \beta_1 X + \varepsilon.
$$

The least-squares estimates choose $\beta_0$ and $\beta_1$ to minimize

$$
\sum_{i=1}^{n}
\left(
y_i - \hat{y}_i
\right)^2.
$$

Coloring observations by cylinder count helps reveal whether different vehicle groups occupy different regions of the plot.

---

## 22. Styled Boxplots

The original boxplot can be enhanced using colors and themes.

```r
ggplot(
  diamonds,
  aes(
    x = cut,
    y = price,
    fill = cut
  )
) +
  geom_boxplot() +
  scale_fill_brewer(
    palette = "Set3"
  ) +
  theme(
    axis.text.x =
      element_text(
        angle = 45,
        hjust = 1
      )
  ) +
  ggtitle(
    "Boxplot of Price by Cut"
  )
```

This example illustrates the distinction between:

- **data mappings**, such as `fill = cut`,
- **plot styling**, such as the palette and label angle.

---

## 23. Scatter Plot with Linear Trend Lines

The relationship between carat and price can be examined separately for each cut category.

```r
ggplot(
  diamonds,
  aes(
    x = carat,
    y = price,
    color = cut
  )
) +
  geom_point(alpha = 0.6) +
  geom_smooth(
    method = "lm",
    se = FALSE,
    linetype = "dashed"
  ) +
  theme_minimal() +
  ggtitle(
    "Price vs Carat with Trend Line"
  )
```

For every group, `geom_smooth(method = "lm")` fits a model of the form

$$
\text{price}
=
\beta_0
+
\beta_1\text{carat}
+
\varepsilon.
$$

The lines make it easier to compare the approximate trends between cut categories.

Setting

```r
se = FALSE
```

removes the confidence band around the fitted regression line.

---

## 24. Density Plot of Price by Cut

```r
ggplot(
  diamonds,
  aes(
    x = price,
    fill = cut
  )
) +
  geom_density(alpha = 0.6) +
  scale_fill_brewer(
    palette = "Paired"
  ) +
  ggtitle(
    "Density Plot of Price by Cut"
  )
```

The transparency allows overlapping group distributions to remain visible.

This plot is useful for comparing the entire shape of the price distributions rather than only their averages.

---

## 25. Faceting

Faceting divides a dataset into multiple panels according to categorical variables.

```r
ggplot(
  diamonds,
  aes(x = price)
) +
  geom_histogram(
    bins = 30,
    fill = "skyblue",
    color = "black"
  ) +
  facet_grid(
    cut ~ clarity
  ) +
  ggtitle(
    "Distribution of Price by Cut and Clarity"
  )
```

Here:

- rows correspond to `cut`,
- columns correspond to `clarity`.

Conceptually, the dataset is divided into subsets

$$
D_{ij}
=
\{
x :
\text{cut}=i
\land
\text{clarity}=j
\}.
$$

A separate histogram is then created for every subset.

Faceting is particularly useful when direct overlays would become too crowded.

---

## 26. Heatmaps

A heatmap represents a third variable through color intensity.

```r
ggplot(
  diamonds,
  aes(
    x = cut,
    y = color,
    fill = carat
  )
) +
  geom_tile() +
  scale_fill_gradient(
    low = "white",
    high = "blue"
  ) +
  ggtitle(
    "Heatmap of Cut and Color vs Carat"
  )
```

The horizontal and vertical positions represent categorical variables, while `fill` represents carat.

The basic mapping is

$$
(\text{cut},\text{color})
\longrightarrow
\text{fill intensity}.
$$

Heatmaps are useful for identifying patterns across combinations of categories.

For aggregated analyses, it is often useful to first calculate a summary such as the mean carat for each combination:

$$
\bar{C}_{ij}
=
\frac{1}{n_{ij}}
\sum_{k=1}^{n_{ij}}
C_{ijk}.
$$

---

## 27. Violin Plots

A violin plot combines ideas from boxplots and density estimation.

```r
ggplot(
  diamonds,
  aes(
    x = cut,
    y = price,
    fill = clarity
  )
) +
  geom_violin() +
  coord_flip() +
  scale_fill_brewer(
    palette = "Set2"
  ) +
  ggtitle(
    "Violin Plot of Price by Cut and Clarity"
  )
```

The width of the violin represents the estimated density of observations at different values.

A wide region indicates relatively high density, while a narrow region indicates relatively low density.

Unlike a boxplot, a violin plot can reveal features such as:

- asymmetry,
- multiple modes,
- concentration of observations.

---

## 28. Multivariate Scatter Plots

Several variables can be represented simultaneously using different aesthetics.

```r
ggplot(
  dsmall,
  aes(
    x = carat,
    y = price,
    size = depth,
    color = cut
  )
) +
  geom_point(alpha = 0.6) +
  scale_size_continuous(
    range = c(2, 10)
  ) +
  ggtitle(
    "Price vs Carat with Point Size by Depth"
  )
```

This plot displays four variables at once:

$$
\begin{aligned}
x &\rightarrow \text{carat},\\
y &\rightarrow \text{price},\\
\text{size} &\rightarrow \text{depth},\\
\text{color} &\rightarrow \text{cut}.
\end{aligned}
$$

This is an example of **multivariate visualization**.

It allows us to investigate whether the carat-price relationship changes according to additional diamond characteristics.

---

# Main Concepts Covered

The notebook introduces several important ideas in exploratory data analysis.

## Data inspection

```r
head()
summary()
str()
nrow()
```

These functions help understand the size, structure, and basic statistical properties of a dataset.

## Random sampling

```r
sample()
set.seed()
```

Sampling makes large datasets easier to explore, while `set.seed()` ensures reproducibility.

## Scatter plots

```r
geom_point()
```

Scatter plots investigate relationships between continuous variables.

## Transformations

```r
log()
```

Logarithmic transformations can reveal relationships that are difficult to see on the original scale.

## Aesthetic mappings

```r
aes(
  x = ...,
  y = ...,
  color = ...,
  shape = ...,
  size = ...,
  fill = ...
)
```

Aesthetic mappings encode variables through visual properties.

## Transparency

```r
alpha = ...
```

Transparency helps reveal density in plots suffering from overplotting.

## Trend estimation

```r
geom_smooth()
stat_smooth()
```

Smooth curves and regression lines summarize relationships between variables.

## Categorical distributions

```r
geom_bar()
```

Bar charts summarize counts of categorical observations.

## Continuous distributions

```r
geom_histogram()
geom_density()
```

Histograms and density plots describe how continuous observations are distributed.

## Distribution comparison

```r
geom_boxplot()
geom_violin()
```

Boxplots and violin plots compare numerical distributions across categories.

## Overlapping categorical observations

```r
geom_jitter()
```

Jitter makes overlapping observations easier to see.

## Faceting

```r
facet_grid()
```

Faceting produces separate plots for combinations of categorical variables.

## Heatmaps

```r
geom_tile()
```

Heatmaps encode numerical information using color intensity.

---

# Overall Exploratory Workflow

The examples illustrate a general EDA process:

$$
\boxed{
\text{Raw Data}
\rightarrow
\text{Inspection}
\rightarrow
\text{Cleaning / Sampling}
\rightarrow
\text{Visualization}
\rightarrow
\text{Pattern Detection}
\rightarrow
\text{Statistical Modeling}
}
$$

For the `diamonds` dataset, an example workflow is:

```text
diamonds
   |
   +--> inspect structure and summaries
   |
   +--> examine carat vs price
   |
   +--> transform variables with log()
   |
   +--> investigate dimensions and outliers
   |
   +--> sample observations
   |
   +--> examine categorical variables
   |
   +--> study distributions
   |
   +--> compare groups
   |
   +--> estimate trends
   |
   +--> construct multivariate visualizations
```

The central principle is that visualization should help answer a question about the data.

For example:

$$
\text{Question}
\rightarrow
\text{variables}
\rightarrow
\text{appropriate visual encoding}
\rightarrow
\text{interpretation}.
$$

Rather than using plots only for presentation, exploratory data analysis uses them as tools for discovering structure, checking assumptions, and deciding what should be investigated next.

The R script and the Jupyter notebook are therefore two different ways of executing and presenting the same analytical workflow:

$$
oxed{
	ext{same R analysis}
ightarrow
egin{cases}
	ext{sequential `.R` script},\\
	ext{interactive `.ipynb` notebook}
\end{cases}
}
$$
