# Apache Spark: End-to-End Industrial IoT Predictive-Maintenance Pipeline

**Portfolio tutorial | PySpark · Spark SQL/DataFrames · Window functions · Parquet · Spark MLlib**

## Project at a glance

This project implements an end-to-end **batch data-engineering and machine-learning pipeline** for tractor telemetry. It transforms noisy daily sensor readings into validated records, builds time-window summaries, integrates maintenance labels, and trains a Random Forest classifier to distinguish records associated with failing and non-failing tractors.

The experiment uses **synthetic data** and runs Spark in Docker/Jupyter using `local[*]` (12-way default parallelism in the reported run). It demonstrates Spark's partitioned execution model **on one machine**, not a deployed multi-node cluster, streaming system, or validated real-world failure forecast.

### Results demonstrated in the project

| Stage | Reported result |
|---|---|
| Data ingestion | 7,500 daily telemetry readings, generated for 250 tractors over 30 days |
| Quality control | 483 corrupted readings filtered out (6.44%); 7,017 retained |
| Feature engineering | Per-tractor, three-calendar-day averages of vibration and heat, with record counts and start/end days |
| Data integration | Cleaned telemetry joined to a tractor-level maintenance registry and saved as Parquet |
| ML experiment | Random Forest tuned by three-fold cross-validation; 5 selected as the reported tree depth |
| Held-out **record-level** test | ROC-AUC **0.9585**; 951 true negatives, 266 true positives, 106 false positives, and 17 false negatives |

**Interpretation:** These are experimental results for the supplied synthetic dataset and split. Because different days from the same tractor can occur in both training and test sets, the result should not be presented as performance on unseen tractors or as a prospective early-warning accuracy.

## 1. Spark concepts illustrated by the implementation

**SparkSession** is the entry point for working with Spark DataFrames and machine-learning pipelines. A **DataFrame** is a distributed, schema-aware table. Its records are organized into **partitions**, the units on which Spark schedules processing tasks. An **RDD** (Resilient Distributed Dataset) is Spark's lower-level distributed collection abstraction; its **lineage** records how a dataset was derived.

**Transformations** such as `filter`, `withColumn`, and `select` describe a computation; Spark ordinarily executes them lazily. **Actions** such as `count`, `show`, and `collect` trigger computation. In the notebook, inspecting `getNumPartitions()` and `toDebugString()` makes partitioning and RDD lineage visible.

The reported session uses:

```python
spark = (SparkSession.builder
         .appName("IndustrialIoTPredictiveMaintenance")
         .master("local[*]")
         .getOrCreate())
```

`local[*]` can use all available local CPU cores, but **does not mean multiple worker machines were deployed**.

## 2. Data ingestion, partitioning, and quality control

The telemetry CSV contains `tractor_id`, `day`, `vibration_index`, `heat_index`, `oil_pressure`, and `sensor_error_code`. Spark's CSV reader loads it with headers and inferred types. The notebook then collects the data into the driver and explicitly parallelizes the rows into **four partitions**, for demonstration. Records marked with a nonzero sensor error code are excluded:

```python
telemetry_clean = telemetry_raw.filter(col("sensor_error_code") == 0)
```

The filter is a **narrow transformation**: each output partition can be produced from its corresponding input partition without redistributing records between partitions. Counting before and after filtering documents the quality-control outcome: **7,500 → 7,017 records**.

**Spark lesson:** The notebook's `collect()` → `sc.parallelize(..., 4)` sequence is useful for illustrating partitions on a small dataset, but it moves all rows through driver memory. It is **not** a scalable ingestion strategy. A larger implementation would keep processing in Spark, choose partitions according to workload, and inspect the execution plan and shuffle costs.

## 3. Feature engineering with Spark Window functions

The pipeline derives fixed, **non-overlapping three-calendar-day groups** for each tractor:

```python
telemetry_marked = telemetry_clean.withColumn(
    "window_id", floor((col("day") - 1) / 3) + 1
)
window_3day = Window.partitionBy("tractor_id", "window_id")
```

Within each tractor/window group, `min(day)`, `max(day)`, `count(*)`, `avg(vibration_index)`, and `avg(heat_index)` generate summary features. One output row is retained per tractor and window using `dropDuplicates(["tractor_id", "window_id"])`.

This captures a central Spark skill: **partition-aware analytical aggregation over groups of records**. Importantly, these are fixed three-day bins, **not** a sliding/rolling window over every consecutive three observations. Because invalid readings were removed first, a window may contain fewer than three valid readings; `records_in_window` exposes that fact.

**Experiment boundary:** The three-day features were created and inspected but **were not inputs to the final Random Forest**, which used the three daily sensor columns. The project therefore demonstrates temporal feature engineering, not a measured improvement from those engineered features.

## 4. Distributed data integration and Parquet

The cleaned telemetry is inner-joined to the maintenance registry on `tractor_id`, adding `crop_type` and `failure_target`. The integrated table is written to **Parquet** and read back for exploration and modelling:

```python
final_telemetry_table = telemetry_clean.join(
    maintenance_registry, on="tractor_id", how="inner"
)
final_telemetry_table.write.mode("overwrite").parquet(output_path)
ml_data = spark.read.parquet(output_path)
```

A **join** combines records using shared keys; unlike the narrow filter, a general join can involve a **shuffle** (data exchange across partitions). **Parquet** is a column-oriented format that supports efficient column-oriented reads and avoids repeating upstream CSV cleaning and joining when a prepared table is reused.

**Spark lesson:** The pipeline combines ingestion, cleaning, integration, and a reusable columnar dataset. Its joins and partition sizes were not benchmarked at distributed-cluster scale.

## 5. Model training with Spark MLlib

Spark MLlib represents features as a single vector column and composes preprocessing and estimation stages into a reusable **Pipeline**:

1. `StringIndexer` produces a numeric `crop_type_index` column.
2. `VectorAssembler` combines **vibration, heat, and oil pressure** into `raw_features`.
3. `StandardScaler` scales that feature vector into `features` (`withMean=False`, `withStd=True`).
4. `RandomForestClassifier` predicts `failure_target` from `features`.

**Important detail:** Although `crop_type_index` was generated, it was **not included** in `VectorAssembler` and thus was not a Random Forest input. The three-day aggregates were also excluded from this model.

The notebook uses an **80/20 random split of telemetry records** (5,677 training; 1,340 testing), and tests Random Forest `maxDepth` values **5, 10, and 15** using a **three-fold `CrossValidator`**. The evaluator selects by **area under the ROC curve (ROC-AUC)**. The full pipeline is fitted inside cross-validation, so its fitted preprocessing stages are part of each fold's model-fitting workflow.

**Spark lesson:** `Pipeline`, `ParamGridBuilder`, `CrossValidator`, and `BinaryClassificationEvaluator` demonstrate how feature preparation, training, and parameter search can be expressed as one Spark ML workflow. Standardizing features is part of the exercise; it is generally not essential for tree split decisions.

## 6. Reading the evaluation correctly

The test confusion matrix contains **951 correctly identified healthy readings** and **266 correctly identified failing readings**, alongside **106 false alarms** and **17 missed failing readings**. The reported ROC-AUC is **0.9585**. On this experimental split, the classifier separates the two record labels well, but the errors matter: a false negative may represent a missed warning, while a false positive may trigger an unnecessary inspection.

The Random Forest's feature-importance plot places **heat index first**, **vibration second**, and **oil pressure far behind**. This is consistent with how the synthetic data generator assigns different heat/vibration distributions to tractor condition profiles, while drawing oil pressure from a profile-independent distribution. **Feature importance indicates predictive association in this model, not a causal mechanism of tractor failure.**

## 7. What would be needed for a stronger real-world evaluation?

These are **recommended extensions, not completed project achievements**:

- **Group-aware validation:** Split by `tractor_id`, so that an entire tractor is held out. The current random record split can place readings from the same tractor in both sets, and the maintenance label is shared by all its readings.
- **Prospective target design:** Define a prediction time and an actual future failure event/horizon; prevent any readings or labels from after the prediction point entering the features.
- **Temporal features in the model:** Test whether the three-day averages improve held-out performance against daily-sensor baselines; handle incomplete windows explicitly.
- **True scale testing:** Avoid driver-side `collect()` during ingestion; measure shuffle, skew, memory, training time, and throughput on larger data and, if needed, a real multi-node Spark deployment.
- **Operational evaluation:** Measure inference latency, calibration, false-alarm cost, and robustness to sensor drift before proposing use on constrained onboard hardware.

## Learning outcomes demonstrated

**Apache Spark data engineering:** Configuring a local PySpark session; reading schema-aware CSV data; understanding partitions, RDD lineage, lazy transformations, and actions; filtering invalid records; using DataFrame and Window APIs; joining datasets; and persisting/reloading Parquet.

**Spark machine learning:** Building MLlib pipelines with categorical indexing, vector assembly, scaling, Random Forest classification, grid-based hyperparameter search, cross-validation, ROC-AUC, confusion matrices, and feature-importance analysis.

**Engineering judgement:** Distinguishing local parallel execution from cluster deployment; recognizing the cost of driver-side collection and shuffles; separating exploratory features from actual model inputs; and identifying why synthetic data, per-tractor label reuse, and record-level splitting limit generalization claims.

## Recruiter-ready summary

> End-to-end PySpark project for synthetic Industrial IoT predictive maintenance: cleaned **7,500 telemetry records**, engineered per-tractor three-day sensor summaries with Spark Window functions, joined maintenance labels, persisted Parquet datasets, and trained/tuned a Spark MLlib Random Forest. The reported **record-level test ROC-AUC was 0.9585**. Demonstrates Spark DataFrames, partitioning and lineage, joins, columnar storage, ML pipelines, and careful interpretation of experimental validation limits.

*Source basis: supplied Spark assessment notebook, synthetic telemetry generator, and accompanying project report. This tutorial covers the Spark pipeline; the archive's separate parallel-KNN/Dask notebook is not part of the reported Spark experiment.*
