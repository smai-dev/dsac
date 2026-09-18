# Descriptive Statistics and Distribution Analysis in R

This document explains the content of both:

- the standalone **R script (`.R`)** containing the same statistical analysis; and
- the corresponding **Jupyter notebook (`.ipynb`)** using an R kernel.

The material introduces **descriptive statistics**, **distribution shape**, and **normality testing** using R.

The main topics are:

- loading census data;
- inspecting a dataset;
- histograms and boxplots;
- mean, trimmed mean, and median;
- standard deviation;
- mode;
- minimum, maximum, and range;
- skewness;
- kurtosis;
- random probability distributions;
- Shapiro-Wilk normality testing;
- Anderson-Darling normality testing;
- Q-Q plots.

The overall workflow is

$$
\text{Load Data}
\rightarrow
\text{Describe Data}
\rightarrow
\text{Visualize Distribution}
\rightarrow
\text{Measure Shape}
\rightarrow
\text{Test Distributional Assumptions}.
$$

---

## 1. R Script and Jupyter Notebook Versions

The same R commands can be executed either in an `.R` file or in an `.ipynb` notebook.

An R script is normally executed sequentially:

$$
C_1 \rightarrow C_2 \rightarrow \cdots \rightarrow C_n.
$$

A Jupyter notebook allows each operation to be run interactively:

$$
\text{execute}
\rightarrow
\text{inspect output}
\rightarrow
\text{continue analysis}.
$$

Because later cells depend on objects created earlier, the notebook should normally be executed from top to bottom.

---

# 2. Loading the Census Dataset

The notebook begins with:

```r
US_census <- read.table(
  "data/uscensus.csv",
  sep = ";",
  header = TRUE
)
```

`read.table()` loads tabular data from an external file.

The argument:

```r
sep = ";"
```

indicates that fields are separated by semicolons.

The argument:

```r
header = TRUE
```

indicates that the first row contains variable names.

The resulting data frame is stored in:

```r
US_census
```

---

# 3. Important Note About R Case Sensitivity

The original notebook then contains:

```r
head(us_census)
```

but the object created in the preceding line is:

```r
US_census
```

R is case-sensitive, so:

```r
US_census
```

and

```r
us_census
```

are different names.

The consistent command is therefore:

```r
head(US_census)
```

This document uses `US_census` consistently.

---

# 4. Inspecting the First Observations

```r
head(US_census)
```

`head()` displays the first rows of the dataset.

This is useful for checking:

- variable names;
- example values;
- whether the data were imported correctly;
- whether columns appear to have sensible formats.

A good exploratory workflow begins with inspecting the raw data before calculating statistics.

---

# 5. Summary Statistics

```r
summary(US_census)
```

`summary()` provides descriptive statistics for each variable.

For a numerical variable $X$, the output commonly includes:

- minimum;
- first quartile $Q_1$;
- median;
- mean;
- third quartile $Q_3$;
- maximum.

These values summarize both the center and spread of the observed distribution.

---

# 6. Number of Observations

```r
nrow(US_census)
```

`nrow()` returns the number of rows in the data frame.

If each row represents one observation, then:

$$
n
=
\texttt{nrow(US\_census)}
$$

is the sample size.

The sample size matters because the reliability of statistical estimates and tests often depends on $n$.

---

# 7. Histogram of Age

```r
hist(US_census$Age)
```

A histogram divides a continuous variable into intervals or **bins**.

For bin $j$ with boundaries $a_j$ and $a_{j+1}$, its frequency is:

$$
n_j
=
\#\{
x_i:
a_j \leq x_i < a_{j+1}
\}.
$$

The histogram of `Age` helps reveal:

- the center of the distribution;
- spread;
- skewness;
- multiple peaks;
- unusual observations.

It provides a graphical view before numerical summaries are interpreted.

---

# 8. Arithmetic Mean

The notebook calculates:

```r
mean(US_census$Age)
```

For ages

$$
x_1,x_2,\ldots,x_n,
$$

the arithmetic mean is:

$$
\bar{x}
=
\frac{1}{n}
\sum_{i=1}^{n}x_i.
$$

The mean describes the average age.

However, it is sensitive to unusually large or small observations.

---

# 9. Trimmed Mean

The notebook then computes:

```r
mean(
  US_census$Age,
  trim = 0.10
)
```

A $10\%$ trimmed mean removes observations from each tail before computing the average.

If the observations are ordered as:

$$
x_{(1)}
\leq
x_{(2)}
\leq
\cdots
\leq
x_{(n)},
$$

a trimmed mean has the general form:

$$
\bar{x}_{trim}
=
\frac{1}{n-2k}
\sum_{i=k+1}^{n-k}
x_{(i)}.
$$

The trimmed mean is less influenced by extreme values than the ordinary mean.

Comparing the two can provide information about whether the tails of the distribution strongly affect the average.

---

# 10. Median

The notebook calculates:

```r
median(US_census$Age)
```

The median is the central value after ordering the observations.

For an odd number of observations:

$$
\tilde{x}
=
x_{\left(\frac{n+1}{2}\right)}.
$$

For an even number of observations, the median is usually the average of the two central observations.

The median is more resistant to extreme values than the arithmetic mean.

Therefore:

$$
\text{mean}
\quad\text{and}\quad
\text{median}
$$

together can provide clues about asymmetry in the distribution.

---

# 11. Mean, Median, and Distribution Shape

For a roughly symmetric distribution:

$$
\text{mean}
\approx
\text{median}.
$$

For a strongly right-skewed distribution, it is common to observe:

$$
\text{mean}
>
\text{median}.
$$

For a strongly left-skewed distribution, it is common to observe:

$$
\text{mean}
<
\text{median}.
$$

These are useful descriptive tendencies, though the complete distribution should also be inspected graphically.

---

# 12. Standard Deviation

The notebook calculates:

```r
sd(US_census$Age)
```

The sample standard deviation measures spread around the sample mean.

It is defined as:

$$
s
=
\sqrt{
\frac{
\sum_{i=1}^{n}
(x_i-\bar{x})^2
}{
n-1
}
}.
$$

The corresponding sample variance is:

$$
s^2
=
\frac{
\sum_{i=1}^{n}
(x_i-\bar{x})^2
}{
n-1
}.
$$

A small standard deviation indicates that observations are relatively concentrated around the mean.

A large standard deviation indicates greater dispersion.

---

# 13. Calculating the Mode

The notebook creates a frequency table:

```r
my_mode <- table(US_census$Age)
my_mode
```

`table()` counts how often each age occurs.

If:

$$
f(x)
=
\text{frequency of value }x,
$$

then the mode is a value satisfying:

$$
x_{mode}
=
\operatorname*{arg\,max}_x f(x).
$$

The notebook identifies the maximum frequency using:

```r
my_mode[
  which(
    my_mode == max(my_mode)
  )
]
```

This can return more than one value if several ages share the highest frequency.

Such a distribution is **multimodal**.

---

# 14. Minimum and Maximum

The notebook calculates:

```r
min(US_census$Age)
max(US_census$Age)
```

The minimum is:

$$
x_{\min}
=
\min_i x_i,
$$

and the maximum is:

$$
x_{\max}
=
\max_i x_i.
$$

Together they describe the observed endpoints of the data.

---

# 15. Range

The notebook also uses:

```r
range(US_census$Age)
```

In R, `range()` returns the minimum and maximum values.

Conceptually:

$$
\operatorname{range}(X)
=
[x_{\min},x_{\max}].
$$

The numerical **range width** is:

$$
R
=
x_{\max}-x_{\min}.
$$

The range is simple to interpret but depends entirely on the two most extreme observations.

---

# 16. Boxplot

The notebook creates:

```r
boxplot(US_census$Age)
```

A boxplot summarizes a distribution using quartiles.

The main values are:

$$
Q_1,
\quad
Q_2,
\quad
Q_3,
$$

where $Q_2$ is the median.

The interquartile range is:

$$
IQR
=
Q_3-Q_1.
$$

A common rule identifies potential outliers below:

$$
Q_1-1.5\,IQR
$$

or above:

$$
Q_3+1.5\,IQR.
$$

The boxplot is useful for quickly examining:

- center;
- spread;
- asymmetry;
- potential outliers.

---

# 17. Loading the `moments` Package

The notebook contains:

```r
#library(e1071)
library(moments)
```

The `e1071` package is commented out.

The `moments` package is loaded and is used to calculate:

- skewness;
- kurtosis.

These are higher-order measures describing the **shape** of a distribution.

---

# 18. Skewness

The notebook computes:

```r
skewness(US_census$Age)
```

Skewness measures asymmetry.

A population-style standardized third central moment can be written as:

$$
\gamma_1
=
\frac{
E[(X-\mu)^3]
}{
\sigma^3
}.
$$

The interpretation is approximately:

$$
\gamma_1 = 0
$$

for a symmetric distribution,

$$
\gamma_1 > 0
$$

for right-skewness, and

$$
\gamma_1 < 0
$$

for left-skewness.

A right-skewed distribution has a longer or heavier right tail.

A left-skewed distribution has a longer or heavier left tail.

The exact finite-sample formula can differ between software implementations, so package documentation should be consulted when exact estimator conventions matter.

---

# 19. Kurtosis

The notebook computes:

```r
kurtosis(US_census$Age)
```

Kurtosis is based on the fourth standardized central moment:

$$
\gamma_2
=
\frac{
E[(X-\mu)^4]
}{
\sigma^4
}.
$$

For the conventional non-excess definition, the normal distribution has kurtosis:

$$
3.
$$

The corresponding **excess kurtosis** is:

$$
\gamma_2 - 3,
$$

for which a normal distribution has value:

$$
0.
$$

Different R packages may report kurtosis using different conventions, so the precise definition used by the selected package matters.

Kurtosis is associated with aspects of tail weight and the concentration of a distribution.

---

# 20. Describing a Distribution Numerically

The statistics calculated so far can be grouped as follows.

### Location

$$
\text{mean},
\quad
\text{median},
\quad
\text{mode}.
$$

### Spread

$$
\text{standard deviation},
\quad
\text{range},
\quad
IQR.
$$

### Shape

$$
\text{skewness},
\quad
\text{kurtosis}.
$$

Together they provide a much richer description than a single average.

---

# 21. Simulating Binomial Data

The notebook sets a random seed:

```r
set.seed(100)
```

This makes the simulated results reproducible.

It then generates:

```r
x <- rbinom(
  15,
  5,
  .6
)
```

`rbinom()` generates random observations from a binomial distribution.

The arguments mean:

- number of observations: $15$;
- number of trials per observation: $5$;
- probability of success per trial: $0.6$.

Thus:

$$
X_i
\sim
\operatorname{Binomial}(5,0.6),
\qquad
i=1,\ldots,15.
$$

---

# 22. Binomial Distribution

For:

$$
X
\sim
\operatorname{Binomial}(n,p),
$$

the probability of observing exactly $k$ successes is:

$$
P(X=k)
=
\binom{n}{k}
p^k
(1-p)^{n-k}.
$$

In this notebook:

$$
n=5
$$

and

$$
p=0.6.
$$

The expected value is:

$$
E[X]
=
np
=
5(0.6)
=
3.
$$

The variance is:

$$
\operatorname{Var}(X)
=
np(1-p)
=
5(0.6)(0.4)
=
1.2.
$$

---

# 23. Inspecting the Simulated Binomial Sample

The notebook displays:

```r
x
```

and plots:

```r
plot(x)
```

This allows the individual simulated observations to be inspected.

Because the binomial distribution is discrete, each value belongs to:

$$
\{0,1,2,3,4,5\}.
$$

With only $15$ observations, the sample is also relatively small.

---

# 24. Shapiro-Wilk Normality Test

The notebook applies:

```r
shapiro.test(x)
```

The Shapiro-Wilk test evaluates the null hypothesis:

$$
H_0:
\text{the data are consistent with a normal distribution}
$$

against:

$$
H_1:
\text{the data are not normally distributed}.
$$

The output includes:

- a test statistic $W$;
- a $p$-value.

A common decision rule at significance level:

$$
\alpha = 0.05
$$

is:

$$
p < \alpha
\Rightarrow
\text{reject }H_0,
$$

while:

$$
p \geq \alpha
\Rightarrow
\text{do not reject }H_0.
$$

Failing to reject $H_0$ does **not** prove that the data are normal.

---

# 25. Why the Binomial Example Is Interesting

The simulated binomial variable is:

- discrete;
- bounded between $0$ and $5$;
- generated from a binomial distribution rather than a normal distribution.

Therefore, it is not theoretically normal.

This example illustrates that normality tests should be interpreted together with knowledge of the data-generating distribution.

Statistical testing should not replace understanding of how the data were generated.

---

# 26. Simulating Lognormal Data

The notebook then generates:

```r
x <- rlnorm(
  20,
  0,
  .4
)
```

This creates $20$ observations from a lognormal distribution.

If:

$$
Y
\sim
N(\mu,\sigma^2),
$$

and:

$$
X=e^Y,
$$

then:

$$
X
$$

has a lognormal distribution.

Here the logarithmic-scale parameters are approximately:

$$
\mu=0
$$

and

$$
\sigma=0.4.
$$

---

# 27. Properties of the Lognormal Distribution

A lognormal random variable satisfies:

$$
X>0.
$$

Its distribution is generally right-skewed.

The transformation:

$$
Y=\log(X)
$$

is normally distributed when $X$ is exactly lognormal.

Therefore:

$$
X \not\sim N(\mu,\sigma^2)
$$

in general, even though:

$$
\log(X)
$$

is normal.

---

# 28. Shapiro-Wilk Test on Lognormal Data

The notebook applies:

```r
shapiro.test(x)
```

and also visualizes the values:

```r
plot(x)
```

Again, the null hypothesis is normality.

Since the simulated sample comes from a lognormal distribution, the underlying distribution is non-normal.

However, statistical tests have sampling variability.

With only:

$$
n=20,
$$

the test may not always detect every departure from normality.

This introduces the important idea of **statistical power**.

---

# 29. Sample Size and Normality Tests

A normality test does not behave independently of sample size.

For small samples:

$$
n \text{ small}
\Rightarrow
\text{limited power against some departures from normality}.
$$

For very large samples:

$$
n \text{ large}
\Rightarrow
\text{very small deviations can become statistically detectable}.
$$

Therefore normality should generally be assessed using both:

- numerical tests;
- graphical diagnostics.

---

# 30. Loading the `nortest` Package

The notebook loads:

```r
library(nortest)
```

This package provides additional tests for normality, including the Anderson-Darling test used later.

---

# 31. Simulating Student's t Data

The notebook generates:

```r
x <- rt(
  500000,
  200
)
```

This creates:

$$
500000
$$

observations from Student's $t$ distribution with:

$$
\nu=200
$$

degrees of freedom.

Symbolically:

$$
X_i
\sim
t_{200}.
$$

---

# 32. Student's t Distribution

Student's $t$ distribution is symmetric around zero but has heavier tails than the standard normal distribution.

As the degrees of freedom increase:

$$
\nu \rightarrow \infty,
$$

the $t$ distribution approaches:

$$
N(0,1).
$$

With:

$$
\nu=200,
$$

the distribution is already quite close to normal, but it is not exactly identical to a normal distribution.

---

# 33. Anderson-Darling Test

The notebook applies:

```r
ad.test(x)
```

The Anderson-Darling test assesses whether a sample is consistent with a specified distribution; here the `nortest` implementation is used as a normality test.

The hypotheses are conceptually:

$$
H_0:
X \text{ follows a normal distribution}
$$

versus:

$$
H_1:
X \text{ does not follow a normal distribution}.
$$

Like other hypothesis tests, the output includes a test statistic and a $p$-value.

---

# 34. Anderson-Darling vs Shapiro-Wilk

Both tests investigate normality, but they use different statistics.

The Shapiro-Wilk test is based on how closely ordered observations correspond to expected normal order statistics.

The Anderson-Darling test is based on differences between an empirical distribution function and a theoretical distribution and gives substantial attention to the tails.

The tests therefore do not necessarily produce identical results for every dataset.

---

# 35. Very Large Samples and Normality Testing

The Student's $t$ example uses:

$$
n=500000.
$$

This is extremely large for a normality test.

Even a small theoretical difference between:

$$
t_{200}
$$

and:

$$
N(0,1)
$$

may become statistically detectable when the sample size is enormous.

This illustrates an important distinction:

$$
\text{statistical significance}
\neq
\text{practical importance}.
$$

A distribution can be visually very close to normal while a formal test detects a statistically significant deviation.

---

# 36. Q-Q Plot

The notebook creates:

```r
qqnorm(x)
```

A normal Q-Q plot compares sample quantiles with theoretical normal quantiles.

If the data are approximately normal, the points should lie approximately along a straight line.

Conceptually, the plot compares:

$$
q_i^{sample}
$$

with:

$$
q_i^{normal}.
$$

A roughly linear pattern supports approximate normality.

Systematic curvature or tail deviations indicate departures from normality.

---

# 37. Adding a Reference Line to a Q-Q Plot

A useful extension is:

```r
qqnorm(x)
qqline(x)
```

`qqline()` adds a reference line to help interpret the plot.

The comparison then becomes visually clearer:

```text
points close to line
        |
        v
approximate normal behavior

systematic departure from line
        |
        v
departure from normality
```

---

# 38. Graphical vs Formal Normality Assessment

A robust workflow combines both approaches:

$$
\boxed{
\text{Histogram}
+
\text{Boxplot}
+
\text{Q-Q Plot}
+
\text{Normality Test}
}
$$

Each provides different information.

### Histogram

Shows the overall shape.

### Boxplot

Highlights spread and potential outliers.

### Q-Q plot

Shows how sample quantiles compare with normal quantiles.

### Formal test

Provides a test statistic and $p$-value.

No single method should be interpreted in isolation.

---

# 39. Hypothesis Testing Framework

The normality tests in the notebook follow the general hypothesis-testing framework.

First specify:

$$
H_0
$$

and:

$$
H_1.
$$

Choose a significance level, often:

$$
\alpha=0.05.
$$

Calculate a test statistic and corresponding $p$-value.

Then:

$$
p<\alpha
\Rightarrow
\text{reject }H_0,
$$

otherwise:

$$
p\geq\alpha
\Rightarrow
\text{do not reject }H_0.
$$

The wording **do not reject** is important.

A large $p$-value is not proof that the null hypothesis is true.

---

# 40. Main Descriptive Statistics

The notebook covers three major aspects of a distribution.

## Center

```r
mean()
median()
table()
```

These help describe where observations are concentrated.

## Spread

```r
sd()
min()
max()
range()
```

These describe variability.

## Shape

```r
skewness()
kurtosis()
```

These describe asymmetry and higher-order distributional characteristics.

---

# 41. Main Probability Distributions Used

The notebook simulates three different distributions.

## Binomial

```r
rbinom()
```

$$
X\sim\operatorname{Binomial}(n,p).
$$

This is discrete and models numbers of successes.

## Lognormal

```r
rlnorm()
```

$$
X=e^Y,
\qquad
Y\sim N(\mu,\sigma^2).
$$

This is continuous, positive, and usually right-skewed.

## Student's t

```r
rt()
```

$$
X\sim t_\nu.
$$

This is continuous, symmetric, and has heavier tails than the normal distribution.

---

# 42. Main R Functions Used

### Reading data

```r
read.table()
```

### Inspecting data

```r
head()
summary()
nrow()
```

### Visualization

```r
hist()
boxplot()
plot()
qqnorm()
```

### Measures of center

```r
mean()
median()
```

### Measures of spread

```r
sd()
min()
max()
range()
```

### Frequencies

```r
table()
```

### Distribution shape

```r
skewness()
kurtosis()
```

### Random simulation

```r
rbinom()
rlnorm()
rt()
```

### Normality testing

```r
shapiro.test()
ad.test()
```

---

# 43. Overall Structure of the Notebook

The analysis follows this progression:

```text
Load census data
       |
       v
Inspect rows and summaries
       |
       v
Histogram of Age
       |
       v
Measures of center
  |       |       |
 mean   median   mode
       |
       v
Measures of spread
  |        |       |
  sd     range   boxplot
       |
       v
Distribution shape
  |             |
skewness     kurtosis
       |
       v
Simulate probability distributions
  |           |          |
binomial   lognormal   Student t
       |
       v
Normality testing
  |                    |
Shapiro-Wilk   Anderson-Darling
       |
       v
Q-Q plot
```

This moves from basic descriptive statistics to formal distribution analysis.

---

# 44. Key Takeaway

The central lesson is that a numerical dataset should not be summarized by a single statistic.

A useful description combines:

$$
\boxed{
\text{Center}
+
\text{Spread}
+
\text{Shape}
+
\text{Visualization}
}
$$

When assumptions such as normality matter, they should be investigated using both graphical and formal methods:

$$
\boxed{
\text{Visual diagnostics}
+
\text{statistical tests}
}
$$

The notebook therefore provides a foundation for later work in:

- statistical inference;
- hypothesis testing;
- regression;
- probability modeling;
- machine learning;
- data science.
