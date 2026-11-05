# Applied Data Science, Machine Learning & Semantic Analysis

This repository collects practical projects in **data preparation, statistical exploration, machine learning, natural language processing, causal semantics, and scalable analytics**.

The emphasis is on complete analytical workflows: understanding the data, preparing features, selecting suitable methods, evaluating results, and interpreting what the evidence does—and does not—support. The projects range from introductory modeling exercises to larger pipelines combining document analysis, semantic comparison, and distributed processing.

## Project Areas

| Area | Focus |
|---|---|
| **Data Preparation & Feature Engineering** | Data quality, missing values, categorical encoding, sampling, feature selection, dimensionality reduction, and time-series preparation |
| **Exploratory Visual Analytics** | Descriptive statistics, statistical graphics, multivariate visualization, and exploratory analysis in R and Python |
| **Classification** | Logistic regression, decision trees, k-nearest neighbors, support vector machines, XGBoost, cross-validation, and classification metrics |
| **Regression** | Linear and multiple regression, Ridge, ElasticNet, support vector regression, residual analysis, and model comparison |
| **Clustering** | K-Means, DBSCAN, standardization, silhouette analysis, and cluster-structure interpretation |
| **Association Pattern Mining** | Apriori, frequent itemsets, support, confidence, lift, leverage, and co-occurrence patterns |
| **End-to-End Predictive Modeling** | CRISP-DM workflow, data understanding, preprocessing, feature engineering, model development, and interpretation |
| **Web Content & Text Mining** | Article collection, text normalization, TF-IDF, topic discovery, dimensionality reduction, and document clustering |
| **Causal & Agentic Semantics** | Global causal-claim alignment, directed semantic relations, evidence-grounded comparisons, and agent-assisted interpretation—without a topic-modeling dependency |
| **Distributed Data Analytics** | PySpark DataFrames, partitioning, aggregations, feature pipelines, and equipment-reliability modeling with Spark MLlib |

## Analytical Approach

Across the projects, the recurring principles are:

- **Inspect before modeling:** understand data provenance, structure, completeness, and distributional characteristics.
- **Make transformations explicit:** document feature preparation, encoding, scaling, and filtering decisions.
- **Evaluate appropriately:** use relevant validation strategies and metrics rather than relying on a single score.
- **Interpret carefully:** distinguish correlation, predictive association, semantic similarity, and causal claims.
- **Keep workflows inspectable:** organize notebooks, supporting data, and explanatory documentation so each analysis can be reviewed and rerun.

## Technologies

**Languages:** Python, R, SQL

**Data and modeling:** pandas, NumPy, scikit-learn, XGBoost, statsmodels, PySpark, Spark MLlib

**Natural language processing:** NLTK, spaCy, TF-IDF, topic modeling, sentence embeddings, and LLM-assisted analysis

**Visualization and notebooks:** Matplotlib, Seaborn, Plotly, ggplot2, Jupyter, VS Code

**Additional tools:** Selenium, Apache Spark, and relevant scientific Python/R packages

## Working with the Projects

Each project is organized as an independent workflow. Depending on the project, its directory may contain Jupyter notebooks, datasets under `data/`, utility modules, a project README, and environment or installation instructions.

To reproduce a workflow:

1. Open the relevant project directory in VS Code or Jupyter.
2. Create or activate the appropriate Python environment, or select an R kernel for R notebooks.
3. Install dependencies from the project's instructions or requirements file, where provided.
4. Check local data paths and any external resources or credentials required by that workflow.
5. Run the notebooks in order and review the newly generated outputs.