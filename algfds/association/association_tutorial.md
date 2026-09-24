# Association Rule Mining — Online Book Purchases

## Project overview

Explore **700 transactions** involving **seven books**, each represented by a binary purchase indicator (1 = purchased; 0 = not purchased). The goal is to discover books commonly purchased together and interpret the resulting patterns for possible recommendations and product bundles.

## Short tutorial

1. **Explore the transactions.** Check the binary features and missing values, then count purchases by book. *Student Social Life Attractions in Dublin* appears in **51.1%** of transactions; *Ethics and Morality* appears in **3.3%**.
2. **Mine frequent itemsets.** Convert purchase indicators to Boolean values and run **Apriori** with a minimum support of **0.10**: retain combinations present in at least 10% of transactions. The notebook identifies **21 frequent itemsets**.
3. **Generate and assess rules.** Form rules of the type `A → B` and examine **support** (frequency of A and B together), **confidence** (fraction of transactions containing A that also contain B), **lift** (co-purchase relative to independence), and **leverage** (observed minus expected co-purchase frequency). The notebook generates **64 rules** at a lift threshold of 1.
4. **Filter and interpret.** Compare rules using the **75th-percentile thresholds** of confidence, lift, and leverage, then inspect examples rather than relying on purchase frequency alone.

## Key findings

- *Introduction to Student Psychology → How to Deal with Procrastination*: **75% confidence** and **1.69 lift** in the reported analysis. Among transactions with the first book, 75% also contain the second; the co-purchase rate exceeds that expected under independence.
- *Student Social Life Attractions in Dublin + Python Programming for Beginners → Introduction to Data Mining*: approximately **70.6% confidence** and **1.95 lift**. This three-book pattern is a candidate for exploring related-item recommendations.

These are **associations, not causal effects** or evidence that a recommendation will increase sales; that would need separate testing.

## Learning outcomes

**Transaction-data preparation · Apriori and frequent itemsets · Association-rule generation · Support/confidence/lift/leverage interpretation · Percentile-based rule filtering · Translating exploratory patterns into testable recommendation ideas**

**Tools:** Python, pandas, NumPy, `mlxtend.frequent_patterns`, Jupyter Notebook.
