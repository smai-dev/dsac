# R Fundamentals: Data Types and Core Data Structures

This document explains the content of both:

- the standalone **R script (`.R`)** containing the same exercises; and
- the corresponding **Jupyter notebook (`.ipynb`)** using an R kernel.

The material introduces the fundamental building blocks of R:

- variables and assignment;
- basic data types;
- vectors;
- lists;
- matrices;
- data frames;
- indexing;
- missing values;
- built-in functions;
- function methods;
- arithmetic summaries such as the mean.

The overall progression is

$$
\text{Values}
\rightarrow
\text{Variables}
\rightarrow
\text{Vectors}
\rightarrow
\text{Lists / Matrices}
\rightarrow
\text{Data Frames}
\rightarrow
\text{Functions}.
$$

---

## 1. R Script and Jupyter Notebook Versions

The same R code can be executed either as an `.R` script or inside an `.ipynb` notebook.

In an R script, statements are normally executed sequentially:

$$
C_1 \rightarrow C_2 \rightarrow \cdots \rightarrow C_n.
$$

In a Jupyter notebook, the code is divided into cells:

$$
\text{execute cell}
\rightarrow
\text{inspect result}
\rightarrow
\text{continue}.
$$

Because later cells reuse variables created earlier, the notebook should normally be executed from top to bottom.

---

# 2. Character Variables

The notebook begins with a character value:

```r
lastName = "Hofmann"

class(lastName)

lastName
```

The assignment

```r
lastName = "Hofmann"
```

creates an object called `lastName`.

Its value is a character string.

The function

```r
class(lastName)
```

returns its class:

```text
"character"
```

Conceptually,

$$
\texttt{lastName}
\mapsto
\text{"Hofmann"}.
$$

A character variable stores textual information.

Examples include:

```r
firstName = "Anna"
city = "Dublin"
course = "Statistics"
```

---

# 3. Numeric Variables

The next cell introduces numerical variables:

```r
numberOfStudents = 20
averageGrade = 67.46

class(numberOfStudents)
class(averageGrade)

numberOfStudents
averageGrade
```

In R, a number written without the integer suffix `L` is normally stored as a numeric value of type `double`.

Therefore,

```r
numberOfStudents = 20
```

and

```r
averageGrade = 67.46
```

are both numerical objects.

Mathematically,

$$
\texttt{numberOfStudents}=20
$$

and

$$
\texttt{averageGrade}=67.46.
$$

The `class()` function allows us to inspect how R represents each object.

---

# 4. Integer and Logical Values

The notebook then introduces an explicit integer:

```r
numberOfStudents = 20L
MSCstudent = TRUE

class(numberOfStudents)
class(MSCstudent)
```

The suffix `L` tells R to create an integer:

```r
20L
```

so

```r
class(numberOfStudents)
```

returns:

```text
"integer"
```

The value

```r
TRUE
```

is logical.

Therefore,

```r
class(MSCstudent)
```

returns:

```text
"logical"
```

Logical values belong to the set

$$
\{\texttt{TRUE},\texttt{FALSE}\}.
$$

They are commonly used in:

- conditions;
- filtering;
- comparisons;
- missing-value checks.

---

# 5. Main Atomic Types Introduced

The first part of the notebook therefore introduces three important R classes:

$$
\boxed{
\text{character},
\quad
\text{numeric/integer},
\quad
\text{logical}
}
$$

For example:

```r
name = "Hofmann"
grade = 74.5
students = 20L
passed = TRUE
```

These values form the basis of larger R data structures.

---

# 6. Creating a Numeric Vector

The notebook creates a vector of student grades:

```r
studentGrades = c(74.5, 85.3, 92.7)
studentGrades
```

The function

```r
c()
```

means **combine** or **concatenate**.

It combines the values into a vector:

$$
\mathbf{g}
=
(74.5,\;85.3,\;92.7).
$$

In R:

```r
studentGrades
```

returns the complete vector.

A vector is one of the most fundamental R data structures.

---

# 7. Vector Indexing

The second element is selected using:

```r
studentGrades[2]
```

R uses **one-based indexing**.

Therefore,

$$
g_1 = 74.5,
$$

$$
g_2 = 85.3,
$$

$$
g_3 = 92.7.
$$

Hence:

```r
studentGrades[2]
```

returns:

```text
85.3
```

This differs from languages such as Python, where indexing begins at zero.

---

# 8. Extending a Vector

The notebook then executes:

```r
studentGrades = c(studentGrades, "100.3")
```

This appends the value `"100.3"` to the existing vector.

However, the new value is written inside quotation marks.

Therefore it is a **character string**, not a numerical value.

R vectors are homogeneous: their elements normally share a common atomic type.

Because the vector now contains a character value, R coerces the numerical elements to character values.

Conceptually,

$$
(74.5,\;85.3,\;92.7)
$$

becomes

$$
(\text{"74.5"},\text{"85.3"},\text{"92.7"},\text{"100.3"}).
$$

This illustrates **type coercion**.

If a numerical grade is intended, it should instead be written as:

```r
studentGrades = c(studentGrades, 100.3)
```

without quotation marks.

---

# 9. Character Vectors

The notebook creates a vector of student names:

```r
lastName = c("Hofmann", "Nolan", "Gray")
lastName = c(lastName, "Tracey")
lastName
```

The result is conceptually

$$
\mathbf{n}
=
(
\text{"Hofmann"},
\text{"Nolan"},
\text{"Gray"},
\text{"Tracey"}
).
$$

A new element can therefore be appended using the same `c()` function.

---

# 10. Indexing Character Vectors

Individual names are accessed with:

```r
lastName[2]
lastName[3]
```

Since R uses one-based indexing:

```r
lastName[2]
```

returns the second name, while:

```r
lastName[3]
```

returns the third.

The general indexing operation is

$$
x[i]
=
\text{the }i\text{-th element of }x.
$$

---

# 11. Lists

The notebook combines two vectors in a list:

```r
resultList = list(
  names = lastName,
  grades = studentGrades
)

resultList
```

Unlike an atomic vector, a list can contain different objects and different data types.

Conceptually,

$$
\texttt{resultList}
=
\{
\texttt{names},
\texttt{grades}
\}.
$$

The list contains two named components:

```text
names
grades
```

This makes lists useful for storing related but structurally different objects.

---

# 12. Accessing List Components

A named component can be selected using `$`:

```r
resultList$names
```

This returns the complete `names` vector.

An element inside that component can then be selected:

```r
resultList$names[2]
```

The access path is therefore:

$$
\texttt{resultList}
\rightarrow
\texttt{names}
\rightarrow
\text{second element}.
$$

This is a common R pattern when working with structured objects.

---

# 13. Matrices

The notebook creates a matrix:

```r
newMatrix = matrix(
  c(1, 2, 3, 4, 5, 6, 9, 11, 10),
  byrow = F,
  ncol = 3
)

newMatrix
```

A matrix is a two-dimensional structure containing elements of a common type.

The supplied vector contains nine elements and the matrix has three columns.

Therefore the number of rows is

$$
\frac{9}{3}=3.
$$

Hence,

$$
M \in \mathbb{R}^{3\times3}.
$$

---

# 14. Filling a Matrix by Columns

The option

```r
byrow = F
```

is equivalent to

```r
byrow = FALSE
```

so R fills the matrix **column by column**.

The input sequence is:

$$
1,2,3,4,5,6,9,11,10.
$$

The resulting matrix is therefore arranged as:

$$
M
=
\begin{bmatrix}
1 & 4 & 9\\
2 & 5 & 11\\
3 & 6 & 10
\end{bmatrix}.
$$

If instead

```r
byrow = TRUE
```

were used, the values would be filled row by row.

---

# 15. Data Frames

The notebook creates a data frame:

```r
studentResultsDataFrame = data.frame(
  names = lastName,
  grades = studentGrades
)

studentResultsDataFrame
```

A data frame is a tabular data structure.

Conceptually,

$$
D
=
\begin{array}{c|c}
\text{names} & \text{grades}\\
\hline
n_1 & g_1\\
n_2 & g_2\\
n_3 & g_3\\
n_4 & g_4
\end{array}.
$$

Each column represents a variable and each row represents an observation.

This is one of the most important structures for statistical analysis in R.

---

# 16. Data Frames vs Matrices

A matrix generally contains one common atomic type.

A data frame can contain columns of different types.

For example:

```r
data.frame(
  name = c("A", "B"),
  grade = c(80, 90),
  passed = c(TRUE, TRUE)
)
```

contains:

- a character column;
- a numerical column;
- a logical column.

Thus:

$$
\text{matrix}
\approx
\text{homogeneous 2D data},
$$

while

$$
\text{data frame}
\approx
\text{heterogeneous tabular data}.
$$

---

# 17. Computing the Mean

The notebook calculates:

```r
mean(studentResultsDataFrame$grades)
```

The `$` operator extracts the `grades` column.

For numerical observations

$$
x_1,x_2,\ldots,x_n,
$$

the arithmetic mean is

$$
\bar{x}
=
\frac{1}{n}
\sum_{i=1}^{n}x_i.
$$

In R:

```r
mean(x)
```

implements this calculation for numerical data.

A useful practical point from the earlier vector example is that a numerical column must actually contain numerical values. If character coercion occurred, the values should be converted or corrected before computing numerical summaries.

---

# 18. Missing Values

The notebook creates a vector containing a missing observation:

```r
studentGrades = c(
  74.5,
  85.3,
  92.7,
  NA
)

studentGrades
```

`NA` means that a value is **not available**.

Conceptually,

$$
\mathbf{g}
=
(74.5,\;85.3,\;92.7,\;NA).
$$

Missing data are common in real datasets.

They may arise because:

- a value was not measured;
- a respondent did not answer;
- a sensor failed;
- data were unavailable;
- a record is incomplete.

---

# 19. Detecting Missing Values

The notebook uses:

```r
is.na(studentGrades)
```

This returns one logical value for every element.

Conceptually,

$$
\operatorname{is.na}(x_i)
=
\begin{cases}
\texttt{TRUE}, & x_i \text{ is missing},\\
\texttt{FALSE}, & \text{otherwise}.
\end{cases}
$$

For the vector in the notebook, the result is equivalent to:

```r
FALSE FALSE FALSE TRUE
```

This is an example of a **vectorized operation**: the function is applied element by element.

---

# 20. Missing Values and Statistical Functions

A missing value normally propagates through many calculations.

For example:

```r
mean(studentGrades)
```

will return `NA` when the vector contains a missing value.

To ignore missing observations intentionally, use:

```r
mean(
  studentGrades,
  na.rm = TRUE
)
```

The mean is then computed from the observed values only:

$$
\bar{x}_{obs}
=
\frac{1}{n_{obs}}
\sum_{i:x_i\neq NA}x_i.
$$

This should be done deliberately because removing missing values changes the set of observations being analyzed.

---

# 21. Inspecting a Function

The notebook evaluates:

```r
mean
```

Entering the name of a function without parentheses allows R to display information about the function object.

This is useful for understanding that functions are themselves objects in R.

For example:

```r
mean
```

refers to the function, while:

```r
mean(x)
```

calls the function on an argument.

---

# 22. Function Methods

The notebook uses:

```r
methods(mean)
```

R supports generic functions.

A generic function can dispatch to different methods depending on the class of its input.

Conceptually,

$$
\text{generic function}
+
\text{object class}
\rightarrow
\text{specific method}.
$$

Thus, a call such as

```r
mean(x)
```

can use an implementation appropriate to the class of `x`.

---

# 23. R Help System

The notebook contains:

```r
?mean
```

The `?` operator opens R's help documentation for a function or topic.

For example:

```r
?mean
```

provides documentation explaining:

- arguments;
- usage;
- behavior;
- examples.

The help system is an essential part of working interactively in R.

---

# 24. Inspecting the Default Method

The notebook uses:

```r
getAnywhere(mean.default)
```

`getAnywhere()` searches for an object even when it is not directly visible in the current environment.

The expression:

```r
mean.default
```

refers to the default method for the generic `mean()` function.

This provides a glimpse into how R's method dispatch works internally.

---

# 25. Creating a Numerical Sequence

The notebook creates:

```r
x <- c(0:10, 50)
```

The expression:

```r
0:10
```

generates the integers

$$
0,1,2,\ldots,10.
$$

Appending `50` produces:

$$
x
=
(0,1,2,\ldots,10,50).
$$

The value `50` is much larger than the other observations and acts as a high extreme value.

---

# 26. Ordinary Mean

The notebook computes:

```r
xm <- mean(x)
```

The ordinary mean is

$$
\bar{x}
=
\frac{1}{n}
\sum_{i=1}^{n}x_i.
$$

Because the value $50$ is much larger than most observations, it pulls the arithmetic mean upward.

This illustrates that the mean can be sensitive to extreme values.

---

# 27. Trimmed Mean

The notebook compares:

```r
c(
  xm,
  mean(x, trim = 0.10)
)
```

A trimmed mean removes a proportion of observations from both tails before calculating the mean.

With:

```r
trim = 0.10
```

approximately $10\%$ of the observations are removed from each end of the ordered sample, subject to R's trimming rule.

If the ordered observations are

$$
x_{(1)}
\leq
x_{(2)}
\leq
\cdots
\leq
x_{(n)},
$$

the trimmed mean is calculated from the central observations rather than the full sample.

Conceptually,

$$
\bar{x}_{trim}
=
\frac{1}{n-2k}
\sum_{i=k+1}^{n-k}
x_{(i)}.
$$

The trimmed mean is therefore less sensitive to extreme observations.

---

# 28. Mean vs Trimmed Mean

The final exercise demonstrates an important statistical distinction.

The ordinary mean uses all observations:

$$
\bar{x}
=
\frac{x_1+\cdots+x_n}{n}.
$$

The trimmed mean deliberately excludes observations from both tails.

Therefore,

$$
\text{ordinary mean}
\quad\text{is more sensitive to outliers},
$$

while

$$
\text{trimmed mean}
\quad\text{is more robust to extreme values}.
$$

This introduces the broader idea of **robust statistics**.

---

# 29. Main R Concepts Covered

The notebook introduces the following concepts.

## Assignment

```r
x = value
x <- value
```

Both forms assign values to objects.

## Inspecting classes

```r
class(x)
```

This identifies the class of an object.

## Vectors

```r
c(...)
```

Vectors store sequences of values.

## Indexing

```r
x[i]
```

This selects the $i$-th element.

## Lists

```r
list(...)
```

Lists can contain several different objects.

## List access

```r
x$name
```

The `$` operator selects named components.

## Matrices

```r
matrix(...)
```

Matrices represent two-dimensional homogeneous data.

## Data frames

```r
data.frame(...)
```

Data frames represent tabular data with potentially different column types.

## Missing values

```r
NA
is.na()
```

These represent and detect unavailable observations.

## Functions

```r
mean()
methods()
getAnywhere()
```

These illustrate function calls and R's method system.

---

# 30. Overall Structure of the Notebook

The notebook progresses from simple values to structured statistical data:

```text
Character values
      |
      v
Numeric / integer values
      |
      v
Logical values
      |
      v
Vectors
      |
      +--> indexing
      |
      +--> type coercion
      |
      v
Lists
      |
      v
Matrices
      |
      v
Data frames
      |
      v
Missing values
      |
      v
Functions and methods
      |
      v
Mean and trimmed mean
```

This progression reflects the basic structure of data analysis in R.

---

# 31. Key Takeaway

The central idea of this exercise is to understand how R represents and organizes data before performing more advanced statistical analysis.

The main hierarchy is:

$$
\boxed{
\text{Atomic Values}
\rightarrow
\text{Vectors}
\rightarrow
\text{Lists / Matrices}
\rightarrow
\text{Data Frames}
}
$$

Once data are stored appropriately, functions such as

```r
mean()
```

can be used to calculate statistical summaries.

Understanding data types, indexing, missing values, and data structures is therefore a prerequisite for later work in:

- exploratory data analysis;
- visualization;
- statistical modeling;
- machine learning.
