# Web Article Mining: Exploration, Topics and Clusters

## Project objective

Explore a supplied corpus of article titles and bodies collected from **spunout.ie**, then compare descriptive text analysis, Latent Dirichlet Allocation (LDA), K-Means, and agglomerative clustering. The analysis uses only the attached corpus and preserves the original measurement, text, label and URL values. This is a text-mining demonstration, not a claim about the audience, effectiveness of advice, or quality of the underlying articles.

## Corpus snapshot

The bundled file `data/youth_information_articles.csv` contains **2,984 rows** and **five renamed fields**. The raw CSV has **nine source-section categories**, **11 absent or `N/A` article bodies**, and **305 blank topic entries**. The analysis notebooks remove unavailable body text before deriving features; the source CSV itself is unchanged except for its first header line. No anonymization or re-scraping was performed.

| New field | Meaning |
|---|---|
| `article_title` | Headline supplied by the source |
| `source_section` | Top-level site category |
| `source_topic` | More detailed site topic, when present |
| `article_body` | Collected article text |
| `source_url` | Original article link |

See `DATA_DICTIONARY.md` for the old-to-new mapping. **The source URL and full article text remain intact**, so the dataset is not anonymized.

## Recommended notebook sequence

| Step | Notebook | Purpose |
|---|---|---|
| 1 | `01_collect_articles.ipynb` | Optional Selenium acquisition; do not rerun if you want the same fixed corpus |
| 2 | `02_explore_article_corpus.ipynb` | Inspect missing text, token counts, section distribution, common vocabulary |
| 3 | `03_lda_topic_discovery.ipynb` | Tokenize, form a dictionary, train LDA, select a topic count using coherence |
| 4 | `04_kmeans_document_clusters.ipynb` | TF-IDF → SVD → unit normalization → K-Means and qualitative cluster interpretation |
| 5 | `05_hierarchical_document_clusters.ipynb` | Compare Ward/Euclidean and complete/cosine linkage and inspect cluster vocabularies |

The notebooks are independent and can be opened directly from the project root. Output graphics are written to `images/`.

## Method and interpretation

**Corpus exploration.** Section frequencies describe which articles were collected, not which issues are most common among young people. Article lengths and word clouds describe the cleaned text, not the full raw text. Site-specific stopwords remove boilerplate and high-frequency expressions; this necessarily shapes the vocabulary that remains.

**LDA topic discovery.** The notebook fits candidate models for 2–15 topics with a fixed random seed and the same `c_v` coherence criterion. The highest-coherence candidate is a starting point, not an objectively correct number of topics. A topic should be named only after inspecting representative articles, not solely its most probable words.

**K-Means.** The parameter choices, TF-IDF weighting, SVD settings, and cluster validation logic have been left intact. Silhouette and Davies–Bouldin scores are internal separation metrics; a high silhouette score does not guarantee meaningful themes. Website section and topic labels are used **after** clustering for description rather than as training targets.

**Agglomerative clustering.** Ward linkage with Euclidean distances and complete linkage with cosine distances emphasize different notions of group separation. Compare both metrics and the leading vocabulary before interpreting any selected group count. Centroid similarity and low-similarity articles can help flag overlapping or heterogeneous clusters.

