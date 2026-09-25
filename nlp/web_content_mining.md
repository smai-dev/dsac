# Web Content Mining: Discovering Themes in Youth Information Articles

**Project type:** Unsupervised learning, web scraping, text mining and topic modelling  
**Domain:** Youth information and wellbeing content  
**Goal:** Discover thematic structure in an article corpus and compare the discovered groupings with the website's editorial taxonomy.

## 1. Project at a glance

A two-pass web collection workflow assembled **2,973 articles** from the SpunOut.ie youth-information website. The corpus had **9 editorial categories and 63 website topics**. Rather than treating those labels as training targets, the study explored the articles as *unlabelled text*, using clustering and topic modelling to identify themes that cross existing category boundaries.

**Demonstrated outcome:** The pipeline recovered distinct groups covering practical recipes, support services, sexual health, rights and general youth experiences. It also exposed the difference between broad document clusters and finer-grained probabilistic topics. This was an analytical prototype, **not** a deployed recommendation system or a study of individual users' behaviour.

## 2. The end-to-end workflow

```text
Website category pages
    → discover and deduplicate article URLs
    → retrieve article metadata and full text
    → structured article corpus (CSV)
    → clean and normalise text
    ├── TF-IDF → Truncated SVD → K-Means / Agglomerative clustering
    └── bag-of-words → Latent Dirichlet Allocation (LDA)
    → inspect keywords, clusters and topics
    → compare discovered groups with editorial categories
```

### A. Collecting the web corpus

**Selenium** automated browsing of dynamically rendered pages. The first pass discovered categories and article links, handled pagination and deduplicated URLs; the second retrieved full article content using explicit waits. URL parsing supplied category/topic metadata. **Pandas** organised the extracted titles, metadata and article text for reproducible analysis.

**Learning outcome:** Design a multi-stage collection process that separates URL discovery from content extraction, and preserve enough metadata to audit later analytical outputs. The report documents this site-specific scraping workflow; it does not establish a general-purpose crawler.

### B. Preparing text and features

The preprocessing pipeline handled missing text, removed URLs/HTML and formatting artefacts, combined titles with body text, removed standard and site-specific stopwords, and used **part-of-speech (POS) tagging with WordNet lemmatisation** to normalise words in context.

For clustering, **TF-IDF** weighted words by their importance within a document relative to the corpus. Its configuration included 1–3 word n-grams, up to 20,000 features, document-frequency filtering, sublinear term frequencies and L2 normalisation. **Truncated SVD** reduced the sparse text representation to **1,988 components retaining 90% of cumulative variance**; vectors were normalised again before clustering.

**Learning outcome:** Understand why text representation and dimensionality reduction matter for sparse, high-dimensional documents. TF-IDF was the chosen representation in this project; the report does **not** establish an experimental performance advantage over all neural embeddings.

## 3. Clustering: two complementary views

**K-Means** assigns each document to the nearest centroid, then updates centroids iteratively. A search over **k = 2–10**, supported by Silhouette and Davies–Bouldin diagnostics, selected **four clusters** in the reported experiment. They captured a broad youth-experience group, information services, text-based support and recipes.

**Agglomerative clustering** repeatedly merges nearby groups to form a hierarchy. Comparing linkage/distance choices led to a **Euclidean-distance, Ward-linkage solution with seven clusters** within the examined range. This representation additionally separated themes such as sexual health and children's rights, while retaining recognisable recipe and crisis-support groupings.

The analysis compared cluster keywords, editorial-category membership, inter-cluster similarity and low-similarity articles. An Irish-language article appearing as a low-similarity case illustrated that a largely English-language lexical pipeline needs multilingual evaluation.

**Learning outcome:** Choose clustering methods by their grouping assumptions, use several diagnostics rather than a single metric, inspect discovered themes against an external taxonomy, and identify representation-sensitive outliers. The four- and seven-cluster solutions answer *different granularity questions*; they are not accuracy scores against ground-truth classes.

## 4. Probabilistic topic modelling

**Latent Dirichlet Allocation (LDA)** models documents as mixtures of topics and topics as distributions over words. Unlike the clustering branch, this stage used a **bag-of-words count representation** and a separately prepared dictionary/corpus rather than TF-IDF document vectors. A search over **2–15 topics** selected **15 topics**, with reported **c_v coherence = 0.5803**.

Topic-word inspection recovered finer themes, including education, employment, recipes, climate, sexual health, mental health and support services. These themes were then compared with the seven hierarchical clusters and the site's original editorial taxonomy.

**Learning outcome:** Explain the difference between assigning a document to a cluster and representing it as a mixture of topics; configure and interpret LDA using topic coherence and representative words, rather than assuming topics are automatically meaningful.

## 5. Project achievements and interpretation

- Built a structured, metadata-preserving collection and text-mining workflow for **2,973 web articles**.
- Produced **four broad K-Means groups**, **seven more specific hierarchical groups**, and **15 LDA topics** under the report's respective selection procedures.
- Showed where model-derived themes corresponded to site topics and where articles grouped by functional purpose across editorial boundaries.
- Identified a concrete multilingual robustness issue through inspection of low-similarity documents.

These findings motivate potential improvements to content navigation or recommendations; **no user-engagement improvement or production recommendation system was measured**.

## 6. Skills and learning outcomes for a technical portfolio

**Data acquisition:** Selenium, dynamic-page handling, pagination, deduplication and structured export.  
**NLP and representation:** text cleaning, POS-aware lemmatisation, stopword design, n-grams, TF-IDF, sparse matrices and Truncated SVD.  
**Unsupervised ML:** K-Means, Ward-linkage agglomerative clustering, LDA, hyperparameter exploration, Silhouette, Davies–Bouldin and topic coherence.  
**Analytical judgement:** distinguish model outputs from editorial labels, inspect clusters and outliers, explain granularity trade-offs, and communicate limitations without equating unsupervised groupings with verified truth.

**Recruiter-facing summary:** End-to-end web text-mining project combining automated article collection, sparse text representations, dimensionality reduction, unsupervised clustering and probabilistic topic modelling to uncover themes and evaluate their alignment with an existing content taxonomy.
