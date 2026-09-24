# Unsupervised Learning: Clustering

**Goal.** Explore an unlabeled, synthetic dataset of 1,500 observations and three numerical features; identify its underlying groups without using class labels.

## Approach

1. **Explore and prepare:** Inspect summary statistics and 2D/3D plots to understand feature distributions and potential groups. Standardize the features before applying distance-based algorithms.
2. **Compare algorithms:** Apply **K-Means**, which assigns observations to cluster centers, and **DBSCAN**, which groups dense regions and can label isolated observations as noise.
3. **Tune and evaluate:** Compare K-Means initializations and cluster counts using visual inspection, the elbow method, silhouette score, and Davies–Bouldin index. Explore DBSCAN's `eps` and `min_samples` using a nearest-neighbor distance plot and parameter experiments.

## Findings

- Visual inspection and the clustering metrics support **seven groups** in this dataset, although one K-Means parameter search returned an eight-cluster solution by splitting an elongated group.
- The default DBSCAN configuration merged most observations into a few groups. With `eps=0.23` and `min_samples=4`, the notebook reports **seven clusters and 73 noise points**.
- **Key lesson:** Parameter-search output alone is not enough to establish a meaningful clustering; compare metrics with cluster shape, separation, and noise handling.

## Learning outcomes

Hands-on practice with **exploratory data analysis, feature scaling, unsupervised model selection, hyperparameter tuning, cluster-quality metrics, and visual interpretation of results without ground-truth labels**.

**Tools:** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, and Plotly.
