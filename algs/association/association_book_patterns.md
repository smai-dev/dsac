# Discovering Book Purchase Associations

## Overview

This project investigates purchasing patterns across **700 online book transactions** involving **seven titles**. Each transaction is represented by binary indicators: `1` denotes a purchased book and `0` denotes an absent book. The analysis uses **Apriori frequent-itemset mining** followed by **association-rule evaluation** to identify titles that tend to appear together in shopping baskets.

The purpose is exploratory. An association can suggest a combination worth testing, but it does not demonstrate that one purchase causes another or that a recommendation would increase sales.

## Dataset and initial inspection

The saved notebook output confirms **700 rows**, **seven integer-valued purchase columns**, and **no missing observations**. The first step examines summary statistics, purchase totals, and a bar chart showing the relative popularity of the titles.

| Book | Recorded purchases | Share of transactions |
|---|---:|---:|
| *Student Social Life Attractions in Dublin* | 358 | 51.14% |
| *How to Deal with Procrastination* | 311 | 44.43% |
| *Python Programming for Beginners* | 259 | 37.00% |
| *Introduction to Data Mining* | 253 | 36.14% |
| *Introduction to Student Psychology* | 140 | 20.00% |
| *Quick Meals for Busy Students* | 61 | 8.71% |
| *Ethics and Morality* | 23 | 3.29% |

Before mining, the workflow removes duplicate column names if present and converts binary purchase flags to Boolean values, as expected by the Apriori implementation.

## Mining and evaluating associations

The analysis proceeds through three complementary steps:

1. **Frequent itemsets:** Apriori identifies combinations with support of at least **0.10**, meaning the books co-occur in 10% or more of the 700 baskets.
2. **Association rules:** The frequent itemsets are transformed into directional rules of the form `antecedent → consequent`. The notebook reports rules using confidence, lift, leverage, and conviction alongside support.
3. **Rule selection:** One comparison applies **75th-percentile thresholds** separately to confidence, lift, and leverage. A second comparison ranks rules by each metric and retains the top 10% (six rules per ranking). These are different selection procedures and should not be conflated.

### How to read the metrics

- **Support:** proportion of all transactions containing both the antecedent and consequent.
- **Confidence:** proportion of transactions containing the antecedent that also contain the consequent.
- **Lift:** observed confidence relative to the consequent's overall purchase frequency. Values above 1 indicate positive association relative to independence.
- **Leverage:** difference between observed joint support and the support expected under independence.
- **Conviction:** a directional measure reflecting how often an antecedent occurs without its consequent relative to the independence baseline.

High confidence may reflect a popular consequent rather than a particularly distinctive pairing; lift and leverage provide additional context.

## Results from the saved notebook outputs

| Quantity | Recorded value |
|---|---:|
| Transactions | 700 |
| Distinct book attributes | 7 |
| Minimum support | 0.10 |
| Frequent itemsets | **21** |
| Generated association rules | **64** |
| 75th-percentile confidence threshold | 0.6607 |
| 75th-percentile lift threshold | 1.8352 |
| 75th-percentile leverage threshold | 0.0611 |
| Rules passing the confidence threshold | 16 |
| Rules passing the lift threshold | 16 |
| Rules passing the leverage threshold | 18 |
| Rules displayed per top-10% ranking | 6 |

### Examples of meaningful patterns

| Association rule | Support | Confidence | Lift | Leverage |
|---|---:|---:|---:|---:|
| *Introduction to Student Psychology* → *How to Deal with Procrastination* | 0.1500 | 0.7500 | 1.6881 | 0.0611 |
| *Python Programming for Beginners* + *Student Social Life Attractions in Dublin* → *Introduction to Data Mining* | 0.1443 | 0.7063 | 1.9542 | 0.0705 |
| *Introduction to Data Mining* → *Python Programming for Beginners* | 0.2186 | 0.6047 | 1.6344 | 0.0848 |

For the first rule, three-quarters of baskets containing *Introduction to Student Psychology* also include *How to Deal with Procrastination*. In the second, the combination of introductory programming and Dublin student-life books is associated with data-mining purchases. The third has the **largest leverage reported in the ranking**: its absolute co-purchase frequency exceeds the independence baseline by approximately 8.48 percentage points.

The highest-ranked rules are not necessarily these three examples. The notebook separately reports a **top confidence of 0.8000** and a **top lift of 2.3384** for more specific multi-book combinations. A rule's ranking depends on the chosen metric, and a high value on one metric need not imply a high value on the others.

## Interpretation and methodological limits

The analysis reveals several related-book combinations that could inform hypotheses for merchandising, cross-selling, or recommendations. In particular, introductory programming and data-mining titles exhibit a notable relationship, while psychology and procrastination titles co-occur with relatively high conditional frequency.

These results must be interpreted within the boundaries of the exercise:

- **Associations are not causal effects.** The notebook contains no intervention or A/B test demonstrating improved outcomes from a suggested bundle.
- **The support threshold excludes uncommon combinations.** For example, less frequently purchased titles may be absent from the final frequent-itemset collection.
- **All rule-selection statistics come from the same transactions.** There is no independent validation set or temporal holdout to measure the stability of the reported patterns.
- **Metric-based rankings answer different questions.** Confidence focuses on conditional frequency, lift emphasizes deviation from independence, and leverage emphasizes absolute excess co-occurrence.

Accordingly, the results are best treated as a set of **candidate relationships for further evaluation**, rather than as a deployment-ready recommendation system.

## Files and reproducibility

The renamed project files from the earlier revision are:

```text
association_book_patterns.ipynb
association_book_patterns.md
data/
    book_transactions.csv
```

**Filename note:** The newly uploaded executed notebook is named `association(1).ipynb` and still refers to `data/student_learning_materials.csv`. The earlier revised project uses `association_book_patterns.ipynb` and `data/book_transactions.csv`. These names must match the version being run; the CSV content itself was preserved in the earlier rename.

Use a Python environment with `pandas`, `numpy`, `matplotlib`, and `mlxtend` to reproduce the workflow. Some `association_rules` calls include a `num_itemsets` argument, so the installed `mlxtend` version should support that signature.

**Execution note:** The figures above come from outputs already saved in the uploaded notebook. This documentation revision did not rerun the pipeline or change any executable code or data.
