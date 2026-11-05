# Clustering: Discovering Structure in Unlabeled Data

## Project objective

This notebook investigates whether a synthetic dataset contains recognizable groups without using reference class labels. It compares **K-Means**, which assigns each observation to a centroid, with **DBSCAN**, which identifies density-connected groups and can leave isolated observations unassigned.

The analysis uses three numerical variables (`att1`, `att2`, `att3`) and combines exploratory visualization, feature standardization, clustering experiments, and internal validation. Cluster counts are interpretations of the observed geometry rather than known ground-truth classes.

## Dataset and exploratory observations

**Input:** `data/synthetic_features.csv`  
**Shape:** 1,500 observations × 3 features  
**Types:** all three columns are `float64`  
**Missing values:** none reported by the saved `data.info()` output

The uploaded notebook includes saved outputs for the initial data preview, completeness check, descriptive statistics, histograms, pairwise scatter plots, and the initial Plotly 3D visualization.

| Attribute | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| `att1` | 1.590 | 2.964 | −4.661 | 7.470 |
| `att2` | −0.031 | 1.750 | −3.896 | 3.764 |
| `att3` | −0.421 | 3.088 | −6.362 | 6.651 |

The notebook's visual interpretation suggests **approximately seven recognizable regions**, with different sizes and shapes. This is a qualitative observation, not a verified number of underlying classes.

## Experimental workflow

1. **Inspect the observations.** Examine feature distributions, two-dimensional projections, and the three-dimensional point cloud.
2. **Standardize the features.** Apply `StandardScaler` so that distance-based models operate on comparably scaled dimensions.
3. **Fit seven-cluster K-Means models.** Compare `init='random'` with `init='k-means++'` and inspect assignments and centroids.
4. **Explore K-Means settings.** Search `n_clusters` in `{6, 7, 8}` alongside initialization, restart count, iteration limit, and random-state choices. A separate specified eight-cluster model is also included.
5. **Inspect clustering quality.** Plot inertia, silhouette score, and Davies–Bouldin index for `k=2` through `10`.
6. **Fit DBSCAN.** Compare its default configuration with a manually selected setting (`eps=0.23`, `min_samples=4`), using a nearest-neighbor-distance plot as an additional diagnostic.

## Model configurations and interpretation

| Experiment | Configuration in notebook | What it examines |
|---|---|---|
| K-Means, random initialization | `n_clusters=7`, `init='random'` | Sensitivity to centroid initialization |
| K-Means++ initialization | `n_clusters=7`, `init='k-means++'` | Centroid placement and resulting partition |
| Alternative K-Means partition | `n_clusters=8`, `init='k-means++'`, `n_init=5`, `max_iter=100`, `random_state=42` | Whether an extra centroid subdivides an elongated region |
| DBSCAN baseline | Library defaults (`eps=0.5`, `min_samples=5`) | Groups and noise under the default neighborhood definition |
| DBSCAN selected setting | `eps=0.23`, `min_samples=4` | Finer density-based separation and noise assignment |

**Seven versus eight groups.** The notebook's written discussion favors roughly seven visually distinct regions, while also considering an eight-centroid partition. Adding centroids can reduce K-Means inertia without necessarily producing more meaningful groups. The silhouette and Davies–Bouldin diagnostics are included to complement visual judgment; the latest uploaded file does **not** contain saved numerical outputs from these diagnostics.

**DBSCAN counts from the notebook narrative.** The notebook text reports **three clusters and two noise points** for the default DBSCAN parameters, and **seven clusters and 73 noise points** for `eps=0.23`, `min_samples=4`. These are previously written observations, **not counts verified from saved DBSCAN execution outputs in the uploaded notebook**.

## Evaluation considerations

- **Inertia** measures within-cluster squared distances and typically decreases as the number of K-Means clusters increases. The K-Means `GridSearchCV` uses its estimator's default score (negative inertia), so its selected configuration should not be interpreted as an independent assessment of the natural cluster count.
- **Silhouette score** assesses cohesion relative to neighboring groups; larger scores are generally more favorable.
- **Davies–Bouldin index** compares within-cluster dispersion against between-cluster separation; smaller scores are generally more favorable.
- **DBSCAN parameter selection** depends on `eps` and `min_samples`. The notebook passes `silhouette_score` directly as `GridSearchCV(..., scoring=silhouette_score)`, but that function does not have the scorer signature required by this interface. Any result from that search should **not** be treated as a validated optimum without correcting the scoring definition.
- **Reproducibility:** the initial seven-cluster K-Means configurations do not set `random_state`, so labels and some partitions may vary between runs.

## Visualization and environment

The notebook generates Matplotlib/Seaborn histograms and scatter plots, plus Plotly three-dimensional views. Its initial code sets `plotly.io.renderers.default = 'png'` to request static Plotly rendering, but the saved 3D output in the uploaded file is Plotly figure data rather than a stored PNG. Depending on the renderer and environment, Plotly may require Kaleido/Chrome for static output or WebGL for interactive display. This affects visualization, not the clustering algorithms or their assignments.

## Reproducibility and saved-output status

Expected project layout:

```text
clustering_structure_discovery.ipynb
clustering_structure_discovery.md
data/
    synthetic_features.csv
```

**Status of the uploaded notebook:** the available saved results establish the dataset properties and initial exploratory visualizations. The later K-Means, quality-metric, and DBSCAN cells have no saved execution outputs in this file. Consequently, this document does **not** claim to have confirmed their final cluster counts, scores, or grid-search selections. The notebook was inspected, **not executed**, while updating this documentation.

The dataset is synthetic, and renaming a file does not by itself constitute data anonymization.
