# Equipment Condition Modelling with PySpark

## Project overview

This project develops a local Apache Spark workflow for exploring synthetic equipment telemetry and identifying machine-level service indicators. The dataset represents **250 fictional agricultural machines** observed over **30 operating days**. Daily vibration, engine-temperature, and hydraulic-pressure measurements are generated using machine-specific baselines, changing workloads, and gradual wear-related patterns.

The study combines Spark DataFrame operations, window calculations, Parquet storage, exploratory visualizations, and a Random Forest classification pipeline. The aim is to demonstrate a reproducible data-engineering and modelling workflow, not to diagnose actual machine faults or make operational maintenance decisions.

**Data provenance:** The measurements and service labels are synthetic. The generator was redesigned to create new values and relationships; this is not just a renaming of columns in the earlier dataset.

## Dataset and attribute definitions

| Source | Records | Description |
|---|---:|---|
| `data/machine_operating_signals.csv` | 7,500 | Daily telemetry from 250 machines across 30 days |
| `data/machine_maintenance_targets.csv` | 250 | One operating environment and simulated service label per machine |

### Daily operating signals

| Attribute | Meaning |
|---|---|
| `machine_id` | Fictional machine identifier used to connect both tables |
| `operating_day` | Day number within the 30-day simulated period |
| `motor_vibration_rms` | Simulated vibration magnitude (mm/s) |
| `engine_temperature_c` | Simulated engine temperature (°C) |
| `hydraulic_pressure_bar` | Simulated hydraulic pressure (bar) |
| `diagnostic_event_code` | Synthetic quality indicator: `0` accepted; `1` and `2` excluded |

### Machine-level attributes

| Attribute | Meaning |
|---|---|
| `machine_id` | Corresponding machine identifier |
| `operating_environment` | Fictional category describing the machine's operating context |
| `maintenance_required` | Synthetic binary service flag, assigned once per machine (`0` or `1`) |

The physical units are illustrative. Neither the generated measurements nor the diagnostic codes represent validated equipment specifications or manufacturer-defined limits.

## Synthetic data generation

The script `data/generate_machine_telemetry.py` creates the two CSV datasets. Its design includes differences between machines, latent wear profiles, shared operating-load effects, time-dependent signal variation, and probabilistically assigned service labels. The supplied files were generated reproducibly with NumPy's random generator and **seed `2026`**.

A change to the seed, modelled distributions, or label-generation rules can alter the resulting dataset and the classifier's eventual performance.

## Processing workflow

1. **Import and inspect.** Read the two CSV files with Spark, infer their schemas, and repartition each input into four Spark-managed partitions.
2. **Filter sensor records.** Retain daily readings with `diagnostic_event_code == 0`; exclude simulated diagnostic events marked `1` or `2`.
3. **Summarize three-day intervals.** For each machine, compute the first and last observed day, accepted-reading count, and mean vibration and temperature in non-overlapping three-day segments. These summaries are exploratory and **are not model inputs**.
4. **Integrate data.** Join accepted daily readings with the machine-level registry on `machine_id`, then write the integrated records to `data/machine_model_records.parquet`.
5. **Visualize signal distributions.** Use a seeded 30% sample of the integrated table to compare simulated engine temperatures by service-label group.
6. **Train a classification pipeline.** Assemble and scale the three numerical sensor measurements, fit a Random Forest, and tune its maximum depth through cross-validation.
7. **Review model outputs.** Examine predictions, a confusion table, ROC-AUC, the ROC curve, the selected depth, and relative feature importances.

### Model configuration

| Setting | Configuration in the notebook |
|---|---|
| Target | `maintenance_required` |
| Numerical predictors | `motor_vibration_rms`, `engine_temperature_c`, `hydraulic_pressure_bar` |
| Feature preparation | `VectorAssembler` followed by `StandardScaler(withMean=False, withStd=True)` |
| Additional transformation | `StringIndexer` for `operating_environment`; its index is **not** included in the classifier's feature vector |
| Estimator | PySpark `RandomForestClassifier`, seed `42` |
| Data split | 80% training / 20% testing, seed `42`, at the **daily-record level** |
| Parameter search | `maxDepth` in `{5, 10, 15}` |
| Validation | Three-fold `CrossValidator`, seed `42` |
| Selection metric | Area under the ROC curve (ROC-AUC) |

## Verified dataset characteristics

The following figures were checked directly against the CSV files included with the revised project. **They are dataset properties, not newly computed Spark training results.**

| Characteristic | Count |
|---|---:|
| Distinct machines | 250 |
| Total daily telemetry records | 7,500 |
| Accepted readings (`diagnostic_event_code = 0`) | 7,060 |
| Excluded readings (`diagnostic_event_code = 1` or `2`) | 440 |
| Machines with `maintenance_required = 1` | 59 |
| Machines with `maintenance_required = 0` | 191 |

### Model results: pending local execution

The latest supplied notebook contains **no saved code-cell outputs or execution counts**. Consequently, it does not establish a new training/test count, ROC-AUC, optimal Random Forest depth, confusion matrix, or feature-importance ranking. These should be reported **only after the revised pipeline has completed successfully** on your virtual machine.

The **0.9585 ROC-AUC** reported by the older project belongs to an earlier, different dataset and must not be presented as a result of this version.

## Interpretation and limitations

The generated labels are associated with latent simulated wear and operating conditions. Predictive patterns may therefore be deliberately stronger or different from those found in real equipment. Correlation and feature importance should not be interpreted as evidence of causality.

The notebook splits **daily observations**, rather than keeping all records for a machine in a single partition. This means that the training and test sets can share machine identities. A record-level ROC-AUC may therefore overstate performance on completely unseen equipment. A machine-grouped split would be required to investigate that separate generalization question.

The three-day summaries are useful for demonstrating temporal aggregation but do not contribute to model training. Likewise, `operating_environment` is indexed by the pipeline but omitted from the assembled predictive features.
