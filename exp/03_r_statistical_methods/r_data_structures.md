# Working with R Objects and Data Structures

## Objective

Practice R's atomic types, vectors, lists, matrices, data frames, missing-value handling, and built-in function inspection using small self-contained examples.

## Guided sequence

1. **Identify scalar types.** Distinguish numeric, integer, and logical values with `class()`.
2. **Build vectors.** Create, extend, and index vectors. Observe that appending a character string to a numeric vector coerces all entries to character.
3. **Represent mixed structures.** Use lists for heterogeneous objects and matrices for two-dimensional homogeneous values.
4. **Construct a data frame.** Combine named and grade vectors, then inspect the resulting column types.
5. **Handle missing data.** Detect `NA` and understand how missing values affect numerical summaries.
6. **Explore generic functions.** Inspect `mean()`, its methods, its help page, and ordinary versus trimmed means.

## Interpretation notes

The `studentGrades` vector becomes character after the string `"100.3"` is added. As a result, a subsequent mean calculation on the grade column is not a valid numeric mean without conversion. This behavior is preserved as an illustration of R's type coercion rather than silently rewriting the experiment.