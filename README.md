# A Big-Data Pipeline for Bank Failure Early Warning

Reproducible codebase, synthetic unit-test suite, and archived aggregate verification evidence for an end-to-end **FDIC Bank Failure Early-Warning** pipeline spanning Call Report acquisition, panel and label engineering, multi-engine processing parity, leak-free feature construction, and cost-sensitive out-of-time evaluation.

**Quick Navigation:** [Route A — Inspect Published Results](#2-route-a--inspect-published-results) | [Environment Setup](#4-environment-setup-documented-windows-workflow) | [Route B — Reproduce the Pipeline](#5-route-b--reproduce-the-pipeline) | [Controlled Final Test](#53-controlled-one-shot-final-held-out-evaluation-step-10) | [Synthetic Tests](#7-synthetic-unit-tests--reproducibility-scope)

---

## 1. Project Overview & Study Scope

This research repository investigates scalable financial-data engineering and out-of-time early-warning modelling across U.S. FDIC-insured commercial and savings institutions around three research questions:

1. **Q1 — Storage Layout & Scalable Processing:** How should four decades of bank reports be stored and processed on a laptop, and when would a Spark cluster be needed? (`benchmark_storage.py`, `src/part3_processing.py`)
2. **Q2 — Data Quality, Survival-Aware Labeling & Leakage Prevention:** What quality problems do the raw data contain, and how does the pipeline handle them without look-ahead bias? (`src/part2_panel.py`, `src/part4_features.py`)
3. **Q3 — Honest Out-of-Time Early-Warning Validation:** Can the pipeline's output support an honestly validated early-warning model? (`src/part5_model.py`, `src/part5_policy_v3.py`, `src/part5_spark_demo.py`)

### Study Scope

- **Data Source:** Public **FDIC BankFind Suite** API (`https://api.fdic.gov/banks`), requiring no API key.
- **Financials Panel:** `1984Q1–2026Q2` (`170` reporting quarters), `1,682,616` bank-quarter observations × `33` fields (`24,064` distinct `CERT` institutions).
- **Historical Event & Institution Tables:** `4,117` historical FDIC `/failures` transaction records (`1934–2026`) and `27,835` `/institutions` records (`4,212` active, `23,623` inactive). Not all `4,117` historical records are confirmed study-period bank failures: in the study period (`>= 1984-03-31`), there are `3,259` transaction rows (`411` assistance rows and `2,848` failure rows covering `2,842` confirmed failure `CERT`s and `3` ambiguous-failure `CERT`s), while `858` earlier pre-1984 records (including `488` historical rows with missing `CERT`) are excluded from study event lookup.
- **Outcome Horizon (`y_fail_next4q`):** Four-quarter forward failure indicator ending `2025Q2` (quarters after `2025Q2` are right-censored for label completeness); merges/voluntary exits are treated as competing exits rather than failures. Individual failed banks may appear in up to four preceding bank-quarter positive observations, so bank-quarter positive counts differ from distinct failed `CERT` counts.
- **Chronological Splits (with Strict One-Quarter Financial Lag):**
  - **Development:** `1984Q1–2007Q4` (first model-eligible quarter `1984Q2` after the 1Q reporting lag; `1,155,374` modelled bank-quarters, `8,623` positive bank-quarters across `2,284` distinct positive `CERT`s).
  - **Purge Gap:** `2008Q1–2008Q4` (`33,070` bank-quarters, excluded from training, imputation, CV, and testing).
  - **Held-Out Test:** `2009Q1–2025Q2` (`387,518` bank-quarters, `1,779` positive bank-quarters across `501` distinct failed `CERT` institutions).
- **Calendar Confirmed Failure Events vs. Model-Label Positive `CERT`s:** Across the local FDIC failure-event table using calendar `FAILDATE` boundaries, normalized `RESTYPE == FAILURE`, exclusion of `ASSISTANCE` records, and exclusion of the `3` conflicting-date ambiguous-failure `CERT`s:
  - **Planned development calendar period (`1984-01-01` through `2007-12-31`):** `2,308` confirmed `FAILURE` events.
  - **Model-eligible development calendar period (`1984-04-01` through `2007-12-31`):** `2,293` confirmed `FAILURE` events.
  - **Held-out test calendar period (`2009-01-01` through `2025-06-30`):** `520` confirmed `FAILURE` events.
  - *Distinction from model labels:* These calendar `FAILDATE` failure counts reflect events occurring inside each date window and differ from the distinct institutions carrying positive forward-looking `y_fail_next4q` model labels in the lagged panels (`2,284` distinct positive development `CERT`s in `1984Q2–2007Q4`; `501` distinct positive held-out test `CERT`s in `2009Q1–2025Q2`) due to boundary `FAILDATE` timing relative to the four-quarter horizon and panel-eligibility requirements.

> **Research Scope Disclaimer:** This project is a reproducible financial-data engineering and out-of-time validation study. It is **not** a production banking supervision or regulatory intervention system.

---

## 2. Route A — Inspect Published Results

All files linked below contain the **archived** aggregate results and metadata from the completed baseline study (not newly generated reproduction outputs). Raw FDIC downloads, bank-level processed Parquets, `reporting_gaps.csv`, `aggregation_result.csv`, `spark_aggregation_result.csv`, fitted local model files, and row-level OOF/test prediction files are deliberately excluded from version control.

For complete table-by-table traceability, original workspace path mappings, and the five-step V3 cryptographic provenance chain, see the [Evidence Traceability Index](EVIDENCE_INDEX.md).

| Stage / Topic | Authoritative Archived Artifacts | What to Inspect |
|---|---|---|
| **Traceability, Config & Git Provenance** | [Evidence Index](EVIDENCE_INDEX.md) · [Core Pins](requirements.txt) · [Spark Pins](requirements-spark.txt) · [Git Attributes](.gitattributes) · [Git Ignore](.gitignore) | Complete audit matrix, exact dependency pins, byte-exact V3 preservation rules, privacy exclusions. |
| **Part 1 — Data Acquisition & Completeness** | [Acquisition Summary](results/part1/acquisition_summary.json) · [Quarter Manifest](results/part1/manifest_financials.json) · [API Reconciliation](results/part1/acquisition_reconciliation_derived.json) | `170/170` complete quarters (`1,682,616` rows, `0` duplicates), `4,117` failures, `27,835` institutions, `10` API anchor checks. |
| **Part 1 — CSV vs. Snappy Parquet Benchmark** | [Storage Summary](reports/benchmark_summary.csv) · [Raw Measurements](reports/benchmark_raw_measurements.csv) · [Benchmark Environment](reports/benchmark_environment.json) | Authoritative 5-round AB/BA 2023-cohort benchmark (`37.2%` space saved; write `4.82x`, full read `2.97x`, 5-col read `4.43x` faster). |
| **Part 2 — Quality, Event Reconciliation & Labels** | [Part 2 Summary (Redacted)](results/part2/part2_summary.json) · [Multi-Event Reconciliation](results/part2/multi_event_reconciliation_appendix_audit.json) · [Decade Missingness](results/part2/missing_by_decade.csv) · [Quality by Year](results/part2/quality_by_year.csv) · [Label Exclusions](results/part2/exclusions_and_invalid.csv) · [Labels by Year](results/part2/labels_by_year.csv) · [Exit-Code Audit](results/part2/exit_code_audit.csv) | `33`-field audit (`P3ASSET` structural start `2001`), competing-risk exclusions, study-period `35` multi-event banks (`31/1/3`). |
| **Part 3 — pandas, Polars & Spark Processing** | [Processing Summary](results/part3/benchmark_summary.csv) · [Paired Runs](results/part3/benchmark_runs.csv) · [3-Engine Parity](results/part3/engine_parity.json) · [Processing Environment](results/part3/environment.json) · [Polars Plan](results/part3/polars_optimized_plan.txt) · [Spark Formatted Plan](results/part3/spark_formatted_plan.txt) · [Spark Extended Plan](results/part3/spark_extended_plan.txt) | Paired 3-round Polars (`0.4983 s`) vs. pandas (`5.8044 s`) medians, `2,373`-group parity across all three engines, sanitized query plans. |
| **Part 4 — Feature Engineering & Leakage Guard** | [Split Audit](results/part4/split_audit.json) · [Feature Schema](results/part4/feature_schema.json) · [Train Missingness](results/part4/train_feature_missingness.csv) · [Pipeline Audit](results/part4/pipeline_audit.json) | `14` 1Q-lagged CAMELS predictors, `2008` purge window verification, train-only `SimpleImputer(median)` + `RobustScaler()` stats. |
| **Part 5 — Temporal Cross-Validation (Provenance)** | [CV Fold Audit](results/part5/cv_fold_audit.csv) · [CV Metrics](results/part5/cv_metrics.csv) · [CV Selection Provenance](results/part5/development_selection.json) | 3 expanding purged folds (`724,466` OOF rows): HistGradientBoosting (`AP=0.16083`) vs. Logistic (`0.09625`). *(Note: `development_selection.json` is retained for CV selection provenance only; its older `0.01597` threshold is superseded by V3 below.)* |
| **Part 5 — V3 Cost-Based Threshold Calibration** | [Authoritative V3 Selection](results/part5/development_selection_v3.json) · [V3 Threshold Audit](results/part5/cost_threshold_audit_v3.json) · [V3 Threshold Sweep](results/part5/cost_threshold_sweep_v3.csv) | Authoritative bank-quarter threshold **`0.015019594386141859`** (`c_fn = $8,639.222k`, `1%` false-alarm ratio) & `0.5%`/`1%`/`5%` sweeps. |
| **Part 5 — Final Held-Out Evaluation (Locked)** | [Final Test Lock](results/part5/final_test_lock.json) · [V3 Test Metrics](results/part5/test_metrics_v3.json) · [V3 Confusion Matrix](results/part5/test_confusion_matrix_v3.csv) | One-shot `2009Q1–2025Q2` evaluation: **`AP = 0.47485`**, **`ROC-AUC = 0.97975`**, `TP=1,624 / FP=8,536 / FN=155 / TN=377,203`. |
| **Part 5 — Spark MLlib Development Demo** | [Spark MLlib Demo Output](results/part5/spark_mllib_demo.json) | Isolated development-only Spark MLlib Logistic Regression (`local[2]`, deterministic `pmod(hash(CERT), 4) == 0`). |

---

## 3. Repository Structure

```text
.
├── .gitattributes                    # Byte-exact LF/raw preservation (text unset) for the 4 V3 provenance files
├── .gitignore                        # Publication-safe exclusions protecting local data, caches, and sensitive CSVs
├── EVIDENCE_INDEX.md                 # Traceability matrix and V3 SHA-256 chain linking evidence files to scripts
├── README.md                         # Project overview, archived evidence links, and reproduction instructions
├── requirements.txt                  # Pinned core Python environment (Python 3.13.7, pandas==3.1.1)
├── requirements-spark.txt            # Separate isolated PySpark environment (pyspark==4.1.4, pandas==2.2.3)
├── fdic_download_starter.py          # Part 1: FDIC BankFind API downloader & completeness checker
├── benchmark_storage.py              # Part 1: 5-round AB/BA CSV vs. Snappy Parquet benchmark (2023 cohort; supports --root)
├── src/
│   ├── part2_panel.py                # Part 2: quality audit, clean bank-quarter panel, 4Q failure labels
│   ├── part3_processing.py           # Part 3: pandas vs. Polars paired benchmark & Spark parity/plans
│   ├── part4_features.py             # Part 4: 1Q-lagged 14 CAMELS ratios, splits & train-only preprocessing
│   ├── part5_model.py                # Part 5: 3-fold expanding purged CV & validation (legacy final test disabled)
│   ├── part5_policy_v3.py            # Part 5: authoritative V3 bank-quarter threshold calibration & locked test runner
│   └── part5_spark_demo.py           # Part 5: isolated development-only local Spark MLlib demonstration
├── tests/
│   ├── test_part2_panel.py           # Part 2 synthetic unittest suite (6 test cases)
│   ├── test_part3_processing.py      # Part 3 synthetic unittest suite (3 test cases)
│   ├── test_part4_features.py        # Part 4 synthetic unittest suite (6 test cases)
│   ├── test_part5_model.py           # Part 5 model/fold synthetic pytest suite (9 test cases)
│   └── test_part5_policy_v3.py       # Part 5 V3 policy & lock-guard synthetic pytest suite (11 test cases)
├── reports/
│   ├── benchmark_environment.json    # Archived Part 1 benchmark system/software metadata
│   ├── benchmark_raw_measurements.csv# Archived 5-round AB/BA raw latency & file-size runs
│   └── benchmark_summary.csv         # Archived Part 1 storage benchmark medians
└── results/
    ├── part1/                        # Archived Part 1 acquisition & API reconciliation summaries
    ├── part2/                        # Archived Part 2 quality, label, exit-code, event & redacted summary files
    ├── part3/                        # Archived Part 3 benchmark metrics, 3-engine parity & sanitized execution plans
    ├── part4/                        # Archived Part 4 split audit, 14-feature schema, missingness & pipeline params
    └── part5/                        # Archived Part 5 CV audits, authoritative V3 policy/test metrics, lock & Spark demo
```

---

## 4. Environment Setup (Documented Windows Workflow)

The supported reproduction workflow is verified on **Windows 11** (`win32`, `Windows-11-10.0.26300-SP0`) with **Python `3.13.7`**. Two separate virtual environments are intentionally maintained because the core analytics stack pins `pandas==3.1.1`, whereas the isolated PySpark environment pins `pandas==2.2.3` for `pyspark==4.1.4` compatibility:

- **Core Environment ([requirements.txt](requirements.txt)):** `pandas==3.1.1`, `numpy==2.5.4`, `pyarrow==26.0.1`, `polars==2.1.0`, `scikit-learn==1.9.1`, `joblib==1.6.0`, `psutil==7.2.2`, `matplotlib==3.11.2`, `pytest==9.2.0`, `requests==2.34.2`.
- **Spark Environment ([requirements-spark.txt](requirements-spark.txt)):** `pyspark==4.1.4`, `py4j==0.10.9.9`, `pandas==2.2.3`, `pyarrow==26.0.1`, `numpy==2.5.4`. Requires a Java Development Kit (`java` on `PATH` and `JAVA_HOME` set).

Run the commands below in **Windows PowerShell** from the **repository root directory** (`src/` uses PEP 420 implicit namespace packages, so `python -m src.<module>` must execute from the repository root). After setup, always invoke the explicit virtual-environment interpreter (`.\.venv\Scripts\python.exe` or `.\.venv_spark\Scripts\python.exe`) rather than relying on an unprefixed `python` command. Cross-platform Linux, macOS, or cloud executions have not been verified in this release.

```powershell
# 1. Verify Python interpreter version (expected: Python 3.13.7)
python --version

# 2. Create and verify the isolated Core environment (.venv)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check

# 3. Create and verify the isolated Spark environment (.venv_spark; requires JDK on PATH / JAVA_HOME)
python -m venv .venv_spark
.\.venv_spark\Scripts\python.exe -m pip install -r requirements-spark.txt
.\.venv_spark\Scripts\python.exe -m pip check
```

### Windows Spark Configuration & Known Runtime Limitations

- **Java/JDK Prerequisite:** PySpark requires a compatible Java Development Kit (`java` on `PATH` and `JAVA_HOME` configured). Note that the exact JDK build used during the original archived experiments was **not** logged in `results/part3/environment.json` or `results/part5/spark_mllib_demo.json`; during later clean-clone dependency and JVM verification on the Windows host, OpenJDK `17.0.21.0.2` was present.
- **Python Worker & Driver Executables (`PYSPARK_PYTHON` / `PYSPARK_DRIVER_PYTHON`):** Because Windows virtual environments provide `python.exe` without `python3.exe`, standalone PySpark sessions can attempt to launch the Windows `python3` alias unless overridden. The repository's Spark modules (`src/part3_processing.py` and `src/part5_spark_demo.py`) already set `os.environ.setdefault("PYSPARK_PYTHON", sys.executable)` and `PYSPARK_DRIVER_PYTHON` internally before building `SparkSession`. When launching interactive or standalone PySpark sessions in PowerShell on Windows (including paths containing spaces), set them explicitly for the current shell session:
  ```powershell
  $env:PYSPARK_PYTHON = (Resolve-Path ".\.venv_spark\Scripts\python.exe").Path
  $env:PYSPARK_DRIVER_PYTHON = $env:PYSPARK_PYTHON
  ```
- **Observed Reproduction Status & Limitation Disclosure:** Original local-mode Spark SQL and Spark MLlib research experiments completed on the study host and produced the archived outputs ([3-Engine Parity](results/part3/engine_parity.json), [Spark Formatted Plan](results/part3/spark_formatted_plan.txt), [Spark Extended Plan](results/part3/spark_extended_plan.txt), [Spark MLlib Demo Output](results/part5/spark_mllib_demo.json)). In subsequent verification on a fresh isolated Windows environment, `requirements-spark.txt` dependency installation (`pip check`) and JVM initialization succeeded, **but standalone local Spark Python-worker runtime checks did not succeed—including a targeted check with explicit `PYSPARK_PYTHON` / `PYSPARK_DRIVER_PYTHON` configuration—and the remaining runtime issue was not conclusively isolated.** Setting `PYSPARK_PYTHON` / `PYSPARK_DRIVER_PYTHON` is **not** claimed to resolve standalone Spark execution on Windows, and neither cross-platform nor fresh end-to-end Spark runtime reproduction has been established.

---

## 5. Route B — Reproduce the Pipeline

All pipeline scripts accept explicit external directories (`--out`, `--raw`/`--panel`/`--labels`/`--results`, or `--root`), keeping freshly downloaded datasets and newly computed outputs cleanly separated from the repository's committed code and archived evidence.

### 5.1 Independent `$WORK_ROOT` & Safe Execution Rules

1. **Always execute from the repository root** (`README.md` directory) so `-m src.<module>` imports resolve correctly.
2. **Choose a new, unused external directory for `$WORK_ROOT`** on every fresh reproduction run. Never point `--root` or `--results` at the repository root in a way that mutates committed archived evidence under `reports/` or `results/` (note that omitting `--root` on `benchmark_storage.py` defaults to the repository root).
3. **Quote `"$WORK_ROOT"` in every command** so paths containing spaces work reliably in PowerShell.
4. **Resumption rule:** Only reuse an existing `$WORK_ROOT` intentionally to resume an interrupted step that supports resumption (specifically `fdic_download_starter.py`, which skips quarters marked `complete: true` in `manifest_financials.json`). Stages guarded against silent overwriting (`src.part5_policy_v3` and `src.part5_spark_demo`) raise `FileExistsError` if their target outputs already exist.
5. **Stop on error:** Do not chain all commands into an uncontrolled "run-all" batch script. If any command fails, stop and inspect the error before proceeding.

Define `$WORK_ROOT` in PowerShell before Step 1:

```powershell
# Run from the repository root (create a fresh external directory outside the repository)
$WORK_ROOT = Join-Path (Split-Path -Parent (Get-Location)) "fdic_repro_work"
New-Item -ItemType Directory -Force -Path $WORK_ROOT | Out-Null
```

If you open a **new** PowerShell terminal later during reproduction, navigate back to the repository root and re-establish the exact same `$WORK_ROOT`:

```powershell
$WORK_ROOT = (Resolve-Path "..\fdic_repro_work").Path
```

### 5.2 Sequential Pipeline Commands (Steps 1–9)

Follow steps strictly in order. For Part 3, **`--benchmark` MUST run before `--spark`**, because `--benchmark` generates `"$WORK_ROOT\results\part3\aggregation_result.csv"` (the reference table required by `--spark`) and initializes `engine_parity.json`.

> **Dependency Note if Spark Fails (Steps 5 or 9):** Stages **6, 7, 8, and 10** (`part4_features`, `part5_model`, and `part5_policy_v3`) read only Part 1 (`failures.parquet`) and Part 2 (`panel/`, `labels/`) outputs rather than Part 3 Spark outputs or `spark_mllib_demo.json`, so an auditor can still verify those downstream Core stages independently if Spark runtime fails locally. However, **skipping an unsuccessful Spark stage does not constitute full end-to-end reproduction** of the assignment.

#### Step 1 — Part 1a: Acquire FDIC Financials (`170Q`), Failures & Institutions
- **Purpose:** Download all raw tables from the public FDIC BankFind API (~1 hour full run due to polite `1.0 s` API paging) and verify quarter completeness (`--start` defaults to `1984-03-31`, `--end` to `2026-06-30`).
- **Environment & Prerequisites:** Core (`.venv`); internet access to `api.fdic.gov`.
- **Command:**
  ```powershell
  .\.venv\Scripts\python.exe fdic_download_starter.py --start 1984-03-31 --end 2026-06-30 --out "$WORK_ROOT\data_raw\fdic"
  ```
- **Expected Outputs & Success Checks:** Creates `"$WORK_ROOT\data_raw\fdic\{financials\REPDTE=YYYYMMDD.parquet, failures.parquet, institutions.parquet, manifest_financials.json}"`. Exit code `0` confirms `duplicate CERT-REPDTE : 0` and `incomplete quarters : none`.
- **Archived Reference:** [Acquisition Summary](results/part1/acquisition_summary.json) · [Quarter Manifest](results/part1/manifest_financials.json)

#### Step 2 — Part 1b: Storage Format Benchmark (CSV vs. Snappy Parquet)
- **Purpose:** Execute the 5-round AB/BA storage layout benchmark on the 2023 Call Report cohort (`18,798 × 33`) and generate comparison charts.
- **Environment & Prerequisites:** Core (`.venv`); Step 1 raw `financials/` under `"$WORK_ROOT\data_raw\fdic"`. *(Always pass `--root "$WORK_ROOT"` to keep repository `reports/` untouched.)*
- **Command:**
  ```powershell
  .\.venv\Scripts\python.exe benchmark_storage.py --root "$WORK_ROOT"
  ```
- **Expected Outputs & Success Checks:** Writes `"$WORK_ROOT\reports\{benchmark_raw_measurements.csv, benchmark_summary.csv, benchmark_environment.json, fig1_storage_size_comparison.png, fig2_read_latency_comparison.png, fig3_benchmark_dashboard.png}"` and logs `Benchmark complete successfully with data equivalence verified.`
- **Archived Reference:** [Storage Summary](reports/benchmark_summary.csv) · [Raw Measurements](reports/benchmark_raw_measurements.csv)

#### Step 3 — Part 2: Data Quality Audit, Clean Panel & 4Q Labels
- **Purpose:** Audit all `33` fields across `170` quarters, reconcile FDIC failure/assistance and institution exit codes without look-ahead distortion, and materialize separate point-in-time `panel/` and `labels/` datasets.
- **Environment & Prerequisites:** Core (`.venv`); Step 1 `"$WORK_ROOT\data_raw\fdic"`.
- **Command:**
  ```powershell
  .\.venv\Scripts\python.exe -m src.part2_panel --raw "$WORK_ROOT\data_raw\fdic" --out "$WORK_ROOT\data_processed\part2" --results "$WORK_ROOT\results\part2"
  ```
- **Expected Outputs & Success Checks:** Creates `"$WORK_ROOT\data_processed\part2\{panel\, labels\, event_lookup_AUDIT_ONLY.parquet}"` and `"$WORK_ROOT\results\part2\{quality_by_year.csv, missing_by_decade.csv, exclusions_and_invalid.csv, reporting_gaps.csv, labels_by_year.csv, exit_code_audit.csv, part2_summary.json}"`. Verify `"duplicate_cert_repdte": 0` in `part2_summary.json`.
- **Archived Reference:** [Part 2 Summary](results/part2/part2_summary.json) · [Decade Missingness](results/part2/missing_by_decade.csv) · [Labels by Year](results/part2/labels_by_year.csv)

#### Step 4 — Part 3a: Paired pandas vs. Polars Benchmark (**Run Before Step 5**)
- **Purpose:** Run 3 paired `(year, state)` aggregation trials comparing pandas and Polars, export the optimized Polars plan, and save the baseline `aggregation_result.csv` required by `--spark`.
- **Environment & Prerequisites:** Core (`.venv`); Step 3 `"$WORK_ROOT\data_processed\part2\{panel,labels}"`.
- **Command:**
  ```powershell
  .\.venv\Scripts\python.exe -m src.part3_processing --benchmark --panel "$WORK_ROOT\data_processed\part2\panel" --labels "$WORK_ROOT\data_processed\part2\labels" --results "$WORK_ROOT\results\part3"
  ```
- **Expected Outputs & Success Checks:** Creates `"$WORK_ROOT\results\part3\{aggregation_result.csv, benchmark_runs.csv, benchmark_summary.csv, engine_parity.json, environment.json, polars_optimized_plan.txt}"` and prints `Pandas/Polars benchmark PASS`. Confirm `pandas_vs_polars.passed` is `true` in `engine_parity.json`.
- **Archived Reference:** [Processing Summary](results/part3/benchmark_summary.csv) · [3-Engine Parity](results/part3/engine_parity.json) · [Polars Plan](results/part3/polars_optimized_plan.txt)

#### Step 5 — Part 3b: PySpark Aggregation Parity & Execution Plans
- **Purpose:** Run the local PySpark (`local[2]`) aggregation on the full labelled panel, verify numerical parity against Step 4's baseline `aggregation_result.csv`, and capture formatted and extended Spark plans.
- **Environment & Prerequisites:** Spark (`.venv_spark`) + JDK; Step 4 `"$WORK_ROOT\results\part3\aggregation_result.csv"`. *(See [Windows Spark limitations](#windows-spark-configuration--known-runtime-limitations).)*
- **Command:**
  ```powershell
  .\.venv_spark\Scripts\python.exe -m src.part3_processing --spark --panel "$WORK_ROOT\data_processed\part2\panel" --labels "$WORK_ROOT\data_processed\part2\labels" --results "$WORK_ROOT\results\part3"
  ```
- **Expected Outputs & Success Checks:** Writes `"$WORK_ROOT\results\part3\{spark_aggregation_result.csv, spark_formatted_plan.txt, spark_extended_plan.txt}"`, sets `engine_parity.json["spark"]["passed"] = true`, and prints `Spark parity PASS`.
- **Archived Reference:** [3-Engine Parity](results/part3/engine_parity.json) · [Spark Formatted Plan](results/part3/spark_formatted_plan.txt) · [Spark Extended Plan](results/part3/spark_extended_plan.txt)

#### Step 6 — Part 4: 1Q-Lagged 14 CAMELS Features & Chronological Splits
- **Purpose:** Join targets at quarter `q` strictly to financial predictors from quarter `q - 1`, enforce the `2008` purge gap (`gap_is_never_modelled: true`), fit `SimpleImputer(median)` + `RobustScaler()` on training data (`1984Q2–2007Q4`) only, and write train/test Parquets plus split/pipeline audits.
- **Environment & Prerequisites:** Core (`.venv`); Step 3 `"$WORK_ROOT\data_processed\part2\{panel,labels}"`.
- **Optional Read-Only Schema Check** *(verified to inspect Parquet footers only and write zero outputs)*:
  ```powershell
  .\.venv\Scripts\python.exe -m src.part4_features --root "$WORK_ROOT" --validate-only
  ```
- **Command:**
  ```powershell
  .\.venv\Scripts\python.exe -m src.part4_features --root "$WORK_ROOT"
  ```
- **Expected Outputs & Success Checks:** Creates `"$WORK_ROOT\data_processed\part4\{train_features.parquet, train_targets.parquet, test_features.parquet, test_targets.parquet, preprocessor_train_only.joblib}"` and `"$WORK_ROOT\results\part4\{split_audit.json, pipeline_audit.json, feature_schema.json, train_feature_missingness.csv}"`. Confirm `"status": "PASS"`, `"test_meets_minimum": true`, and `"gap_is_never_modelled": true`.
- **Archived Reference:** [Split Audit](results/part4/split_audit.json) · [Feature Schema](results/part4/feature_schema.json) · [Pipeline Audit](results/part4/pipeline_audit.json)

#### Step 7 — Part 5a: 3-Fold Expanding Purged Development CV (`SEED = 42`)
- **Purpose:** Evaluate `LogisticRegression` and `HistGradientBoostingClassifier` (`random_state=42`) across 3 expanding purged cross-validation folds inside development data without reading `test_features.parquet` or `test_targets.parquet`, and save out-of-fold predictions plus CV audits.
- **Environment & Prerequisites:** Core (`.venv`); Step 6 Part 4 development files and Step 1 `"$WORK_ROOT\data_raw\fdic\failures.parquet"`.
- **Optional Read-Only Schema Check** *(verified to inspect schemas of development Parquets and `failures.parquet` only without reading test data or launching training)*:
  ```powershell
  .\.venv\Scripts\python.exe -m src.part5_model --phase validate --root "$WORK_ROOT"
  ```
- **Command:**
  ```powershell
  .\.venv\Scripts\python.exe -m src.part5_model --phase development --root "$WORK_ROOT"
  ```
- **Expected Outputs & Success Checks:** Creates `"$WORK_ROOT\results\part5\{cv_metrics.csv, cv_fold_audit.csv, development_selection.json, cost_threshold_sweep.csv, cost_threshold_audit.json}"` and `"$WORK_ROOT\data_processed\part5\chosen_model_cv_predictions.parquet"`. Confirm `"status": "DEVELOPMENT_LOCKED_TEST_UNTOUCHED"` and `"test_data_read": false`. *(Legacy threshold outputs generated in this step are superseded by Step 8.)*
- **Archived Reference:** [CV Fold Audit](results/part5/cv_fold_audit.csv) · [CV Metrics](results/part5/cv_metrics.csv) · [CV Selection Provenance](results/part5/development_selection.json)

#### Step 8 — Part 5b: Authoritative V3 Cost-Based Threshold Recalibration
- **Purpose:** Calibrate the bank-quarter `(CERT, REPDTE)` alarm threshold solely from Step 7's frozen OOF predictions and `1984–2007` positive `COST` records in `failures.parquet`, without retraining CV or reading held-out test data.
- **Environment & Prerequisites:** Core (`.venv`); Step 7 outputs and `"$WORK_ROOT\data_raw\fdic\failures.parquet"`; no pre-existing `*_v3.*` outputs or `final_test_lock.json` under `"$WORK_ROOT\results\part5\"`.
- **Command:**
  ```powershell
  .\.venv\Scripts\python.exe -m src.part5_policy_v3 --phase recalibrate --root "$WORK_ROOT"
  ```
- **Expected Outputs & Success Checks:** Creates `"$WORK_ROOT\results\part5\{cost_threshold_sweep_v3.csv, cost_threshold_audit_v3.json, development_selection_v3.json}"` embedding SHA-256 hashes of `development_selection.json`, `cv_fold_audit.csv`, and `chosen_model_cv_predictions.parquet`. Confirm `"cv_retrained": false` and `"test_data_accessed": false`.
- **Archived Reference:** [Authoritative V3 Selection](results/part5/development_selection_v3.json) · [V3 Threshold Audit](results/part5/cost_threshold_audit_v3.json) · [V3 Threshold Sweep](results/part5/cost_threshold_sweep_v3.csv)

#### Step 9 — Part 5c: Isolated Development-Only Spark MLlib Demonstration
- **Purpose:** Demonstrate an isolated local Spark MLlib `Pipeline` (`Imputer`, `VectorAssembler`, `StandardScaler`, `LogisticRegression`) on the `pmod(hash(CERT), 4) == 0` sample of Part 4 training data (`<= 1998Q4` train, `1999` purged, `2000Q1–2007Q4` validation).
- **Environment & Prerequisites:** Spark (`.venv_spark`) + JDK; Step 6 Part 4 development outputs (`feature_schema.json`, `train_features.parquet`, `train_targets.parquet`); `spark_mllib_demo.json` must not already exist under `"$WORK_ROOT\results\part5\"`. *(See [Windows Spark limitations](#windows-spark-configuration--known-runtime-limitations).)*
- **Command:**
  ```powershell
  .\.venv_spark\Scripts\python.exe -m src.part5_spark_demo --root "$WORK_ROOT"
  ```
- **Expected Outputs & Success Checks:** Creates `"$WORK_ROOT\results\part5\spark_mllib_demo.json"`. Confirm `"status": "PASS_SPARK_DEVELOPMENT_ONLY"` and `"heldout_test_read": false`.
- **Archived Reference:** [Spark MLlib Demo Output](results/part5/spark_mllib_demo.json)

### 5.3 Controlled One-Shot Final Held-Out Evaluation (Step 10)

The held-out evaluation (`2009Q1–2025Q2`) is governed by strict one-shot leakage protection:

- **Authoritative Evaluator Only:** V3 (`src.part5_policy_v3`) is the sole authoritative threshold calibration and held-out evaluator. The legacy `src.part5_model --phase final-test` path is permanently disabled (`PermissionError`).
- **Pre-Evaluation Lock & Verification:** Running `src.part5_policy_v3 --phase final-test --root "$WORK_ROOT" --confirm-final-test` requires a fresh, previously unevaluated `$WORK_ROOT` where Steps 6–8 succeeded. Before reading `test_features.parquet`, it verifies SHA-256 hashes against `development_selection_v3.json`, refits `hist_gradient_boosting` (`SEED = 42`) on development `train` data only, and exclusively creates `"$WORK_ROOT\results\part5\final_test_lock.json"` (`"x"` mode).
- **Immutable Lock Policy:** **Never** delete, reset, edit, or bypass [final_test_lock.json](results/part5/final_test_lock.json) in this repository (or in a locked `$WORK_ROOT`), and **never** use held-out test metrics to revisit features, hyperparameters, model selection, or decision thresholds.
- **Usage Scope:** Provided exclusively for authorized independent reproduction on an external `$WORK_ROOT`. Running Step 10 against the repository root is rejected automatically (`FileExistsError` / lock detection) to protect the archived held-out record.

```powershell
# CONTROLLED ONE-SHOT STEP — Execute ONLY once on a fresh external $WORK_ROOT after Steps 6–8 succeed:
.\.venv\Scripts\python.exe -m src.part5_policy_v3 --phase final-test --root "$WORK_ROOT" --confirm-final-test
```

- **Expected Outputs & Success Checks:** Writes `"$WORK_ROOT\results\part5\{final_test_lock.json, test_metrics_v3.json, test_confusion_matrix_v3.csv}"` and `"$WORK_ROOT\data_processed\part5\heldout_predictions_v3_LOCAL_ONLY.parquet"`. Confirm `"status": "V3_FINAL_TEST_COMPLETE_LOCKED_NO_REPEATS"` and `"no_repeats": true`.
- **Archived Reference:** [Final Test Lock](results/part5/final_test_lock.json) · [V3 Test Metrics](results/part5/test_metrics_v3.json) · [V3 Confusion Matrix](results/part5/test_confusion_matrix_v3.csv)

---

## 6. Key Archived Results & Verification Criteria

### 6.1 Summary of Archived Empirical Findings (Parts 1–5)

- **Storage Format (Q1, 2023 Cohort `18,798 × 33`):** Snappy Parquet reduced size by `37.2%` (`2.8138 MB` vs. `4.4828 MB` CSV) with faster median write (`0.0744 s` vs. `0.3582 s`, `4.82x`), full read (`0.0195 s` vs. `0.0579 s`, `2.97x`), and 5-column read (`0.0045 s` vs. `0.0200 s`, `4.43x`). ([Storage Summary](reports/benchmark_summary.csv))
- **Processing Benchmark & Parity (Q1, `1,599,071` labelled rows → `2,373` year-state groups):** Paired 3-round medians showed Polars (`0.4983 s`, `502.61 MiB` peak Python RSS) vs. pandas (`5.8044 s`, `692.60 MiB` peak Python RSS). Cross-engine equivalence passed across pandas, Polars (`0.0` max abs diff), and Spark (`9.97466e-17` max `positive_label_rate` diff). These single-machine measurements show that columnar Polars handles the full four-decade FDIC panel comfortably on a laptop; a Spark cluster becomes justified when datasets exceed single-node memory/disk or require multi-node distributed ETL and training. ([Processing Summary](results/part3/benchmark_summary.csv), [3-Engine Parity](results/part3/engine_parity.json))
- **Data Quality & Study-Period Events (Q2):** All `1,682,616` panel rows (`170` quarters) retained with `0` duplicate `(CERT, REPDTE)` keys (`33` reporting gaps audited). From `3,259` study-period failure/assistance transaction rows (`>= 1984-03-31`), `35` multi-event-date institutions reconcile cleanly (`31` ASSISTANCE+FAILURE, `1` multiple ASSISTANCE-only, `3` conflicting FAILURE dates); full `1934–2026` history (`44/37/4/3`) is preserved separately in [Multi-Event Reconciliation](results/part2/multi_event_reconciliation_appendix_audit.json). ([Part 2 Summary](results/part2/part2_summary.json))
- **Feature Pipeline (Q2):** `14` CAMELS predictors lagged by `1` quarter; train-only `SimpleImputer(median)` + `RobustScaler()`. Development split (`1984Q2–2007Q4`): `1,155,374` rows (`8,623` positives); `2008` purge gap: `33,070` rows; held-out test (`2009Q1–2025Q2`): `387,518` rows (`1,779` positives, `501` distinct failed `CERT`s). ([Split Audit](results/part4/split_audit.json), [Feature Schema](results/part4/feature_schema.json))
- **Development Selection & Authoritative V3 Policy (Q3):** Across `3` purged expanding folds (`724,466` OOF rows, `1,791` positives, `SEED = 42`), HistGradientBoosting achieved higher mean fold AP (`0.16083`) than Logistic Regression (`0.09625`). V3 bank-quarter calibration (`2,237` positive-`COST` `1984–2007` failures → median `COST = $8,639.222k`; assumed `1%` false-alarm cost `$86.39222k`) set threshold **`0.015019594386141859`** (OOF `TP=1,521`, `FP=15,007`, `FN=270`, `TN=707,668`). *(Legacy `development_selection.json` threshold `0.01597` is superseded by V3.)* ([Authoritative V3 Selection](results/part5/development_selection_v3.json), [V3 Threshold Audit](results/part5/cost_threshold_audit_v3.json))
- **Locked Held-Out Evaluation (`2009Q1–2025Q2`, One-Shot V3 Result):** **`AP = 0.47485`**, **`ROC-AUC = 0.97975`**, **`Recall = 0.91287`**, **`Precision = 0.15984`**, **`F1 = 0.27205`**, **`TN = 377,203 / FP = 8,536 / FN = 155 / TP = 1,624`**, 1% proxy cost `$2,076,523.40k` (`$5,358.52k` per `1,000` bank-quarters). Separate development-only Spark MLlib demo (`206,850` train / `71,443` val): `areaUnderPR = 0.03022`, `areaUnderROC = 0.88324`. ([V3 Test Metrics](results/part5/test_metrics_v3.json), [Spark MLlib Demo Output](results/part5/spark_mllib_demo.json))

### 6.2 Reference vs. Fresh-Run Verification Criteria

When comparing newly generated outputs under `"$WORK_ROOT"` against the archived reference files in `reports/` and `results/`, evaluate three distinct categories:

| Category | What Is Expected | Comparison Guidance |
|---|---|---|
| **Structural Invariants** | `14` CAMELS feature names ([Feature Schema](results/part4/feature_schema.json)), strict `q - 1` financial lag, `y_fail_next4q` label definition, `2008` purge window (`gap_is_never_modelled: true`), train-only `SimpleImputer` + `RobustScaler` fit, fixed `SEED = 42` (`part5_model.py` / `part5_policy_v3.py`), deterministic `pmod(hash(CERT), 4) == 0` sampling (`part5_spark_demo.py`), and `final_test_lock.json` governance. | Must hold exactly across every valid reproduction run. |
| **Snapshot-Dependent Results** | Data row counts, label counts, multi-event reconciliation counts, CV metrics, calibrated V3 threshold, and held-out V3 metrics (`AP`, `ROC-AUC`, confusion matrix). | Expected to agree closely under equivalent data and package pins. However, because Step 1 downloads from the live FDIC BankFind API, historical rows, `FAILDATE`, or retrospective `COST` values can change if the FDIC updates records over time; investigate actual raw diffs rather than assuming live-API drift without checking. |
| **Environment-Dependent Measurements** | Raw wall-clock seconds, speedup multipliers, sampled Python-process RSS MiB, and low-level `AdaptiveSparkPlan` / hardware execution-plan details. | Expected to vary across CPU architecture, disk I/O speed, memory capacity, OS file caching, and JVM builds; verify directional conclusions and cross-engine numerical parity rather than matching exact archived timings. |

---

## 7. Synthetic Unit Tests & Reproducibility Scope

All `35` unit tests under `tests/` (`test_part2_panel.py`: 6, `test_part3_processing.py`: 3, `test_part4_features.py`: 6, `test_part5_model.py`: 9, `test_part5_policy_v3.py`: 11) use **synthetic in-memory or temporary-directory fixtures only**. None access raw/processed FDIC Parquets, live network APIs, or `results/part5/final_test_lock.json`.

Execute from the repository root inside the Core environment (using `-B -p no:cacheprovider` avoids creating untracked cache artifacts; standard `.\.venv\Scripts\python.exe -m pytest tests -v` also runs the full suite):

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests -v -p no:cacheprovider
```

During prior isolated same-machine clean-clone verification of the executable codebase commit, all **`35` synthetic unit tests passed in `9.44s`** (`0` failed, `0` skipped) and all `8` CLI `--help` entry points succeeded.

> **Test Scope Boundary:** Passing the 35 synthetic unit tests verifies transformation logic, leakage guards, purged CV splitting, V3 threshold logic, and lock invariants. It does **not** constitute end-to-end execution against live FDIC BankFind downloads or the full `1.68M`-row panel, nor were runtime tests rerun after later documentation-only edits.

---

## 8. Limitations & Methodological Boundaries

1. **Period-End Date vs. Filing Release Vintage:** `REPDTE` reflects quarter-end reporting dates rather than historical Call Report release timestamps; the one-quarter lag reduces look-ahead bias without providing a formal real-time publication guarantee.
2. **Retrospective Cost Proxy:** FDIC `COST` reflects retrospective resolution-loss estimates (not inflation-adjusted observed intervention savings), and false-alarm review costs (`0.5%`, `1%`, `5%`) are explicit scenario assumptions charged per bank-quarter rather than per unique failure event.
3. **Hardware & Timing Variance:** Benchmark timings and memory footprints (`reports/benchmark_*`, `results/part3/benchmark_*`) depend on local CPU, storage, OS cache state, and library builds.
4. **Unscripted Historical Part 1 Audit Summaries:** `fdic_download_starter.py` directly generates `manifest_financials.json`, whereas archived aggregate reconciliation files ([Acquisition Summary](results/part1/acquisition_summary.json) and [API Reconciliation](results/part1/acquisition_reconciliation_derived.json)) do not have standalone `.py` generator scripts in the codebase.
5. **Unpublished OOF Hash Target:** [Authoritative V3 Selection](results/part5/development_selection_v3.json) records `source_oof_sha256` (`13a6d6da...`) for row-level OOF predictions (`chosen_model_cv_predictions.parquet`), which remain excluded from public release to avoid publishing bank-level `CERT` predictions; that digest can only be recomputed locally after running Part 5 development (Step 7).
6. **Local Spark Execution Scope:** Archived Spark plans (`AdaptiveSparkPlan isFinalPlan=false`) and `local[2]` runs illustrate distributed APIs on a single machine rather than multi-node cluster scaling, and fresh Windows standalone Spark worker execution remains unresolved (see Section 4).
7. **Sanitized & Relocated Public Paths:** Relative links under `reports/` and `results/` reflect publication-safe relocations and redactions (`part2_summary.json` masks the `3` ambiguous `CERT` integers; Part 3 plans replace personal prefixes with `<PROJECT_ROOT>`), fully documented in [EVIDENCE_INDEX.md](EVIDENCE_INDEX.md).

---

## 9. References & Academic Attribution

- **Data Source & Schema Documentation:** Federal Deposit Insurance Corporation (FDIC) BankFind Suite API (`https://api.fdic.gov/banks/docs/`) and RIS View Field Properties (`https://api.fdic.gov/banks/docs/risview_properties.yaml`).
- **Course Starter Code & Software Ecosystem:** Course-provided starter script (`fdic_download_starter.py`) extended across Parts 1–5 using Python, `pandas`, `numpy`, `pyarrow`, `polars`, `scikit-learn`, `joblib`, `psutil`, `matplotlib`, `requests`, `pytest`, and Apache `pyspark`.
- **AI Assistance Attribution:** **ChatGPT** and the **Google Antigravity coding agent** assisted with pipeline structuring, code development, debugging, unit-test authoring, evidence cross-checking, technical documentation, review, report editing, and translation under human direction and verification.
