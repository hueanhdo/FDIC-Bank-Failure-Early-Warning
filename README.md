# A Big-Data Pipeline for Bank Failure Early Warning

Reproducible codebase, synthetic test suite, and aggregate verification evidence for an end-to-end **FDIC Bank Failure Early-Warning** pipeline spanning Call Report acquisition, panel & label engineering, multi-engine processing parity, leak-free financial feature construction, and cost-sensitive out-of-time evaluation.

---

## 1. Project Overview

This research repository investigates scalable financial-data engineering and out-of-time early-warning modelling across U.S. FDIC-insured commercial and savings institutions around three research themes:

1. **Q1 — Storage Layout & Analytical Processing Efficiency:** Quantifying storage compaction and read/write performance of columnar Snappy Parquet versus CSV (`benchmark_storage.py`), and comparing single-machine pandas, Polars, and local PySpark SQL aggregations under strict numerical parity (`src/part3_processing.py`).
2. **Q2 — Data Quality, Survival-Aware Labeling & Leakage Prevention:** Auditing 170 quarters of historical Call Reports, reconciling FDIC failure/assistance and institution exit records without competing-risk distortion (`src/part2_panel.py`), and building 14 CAMELS-inspired financial predictors with a strict one-quarter reporting lag and train-only preprocessing (`src/part4_features.py`).
3. **Q3 — Out-of-Time Early-Warning Modelling & Decision Policy:** Evaluating regularized Logistic Regression and Histogram-based Gradient Boosting across 3 expanding purged cross-validation folds (`src/part5_model.py`), calibrating a bank-quarter cost-sensitive decision threshold (`0.015019594386141859`) under an explicit false-alarm cost assumption (`src/part5_policy_v3.py`), demonstrating an isolated local Spark MLlib pipeline (`src/part5_spark_demo.py`), and conducting a one-time locked held-out evaluation.

### Study Scope
- **Financials Panel:** `1984Q1–2026Q2` (`170` quarters), `1,682,616` bank-quarter observations × `33` fields (`24,064` distinct `CERT` institutions).
- **Event & Institution Metadata:** `4,117` historical FDIC `/failures` records (`1934–2026`) and `27,835` `/institutions` records (`4,212` active, `23,623` inactive).
- **Temporal Splits (with 1Q financial lag & 4Q outcome horizon `y_fail_next4q` ending `2025Q2`):**
  - **Development:** `1984Q1–2007Q4` (first model-eligible quarter `1984Q2` after 1Q lag; `1,155,374` modelled bank-quarters).
  - **Purge Gap:** `2008Q1–2008Q4` (`33,070` bank-quarters, excluded from training and testing).
  - **Held-Out Test:** `2009Q1–2025Q2` (`387,518` bank-quarters, `1,779` positive observations across `501` failed institutions).

> **Research Scope Disclaimer:** This project is designed strictly as a reproducible financial-data engineering and out-of-time validation study. It is **not** a production banking supervision or regulatory intervention system.

---

## 2. Repository Structure

This public release excludes all raw FDIC datasets, bank-level processed Parquets, row-level predictions, fitted model artifacts, and local caches. Only verified source code, synthetic unit tests, environment specs, and aggregate/sanitized evidence are included:

```text
.
├── .gitignore                        # Publication-safe exclusions protecting local data, caches, and sensitive CSVs
├── EVIDENCE_INDEX.md                 # Traceability matrix linking report/Appendix claims to verified public files
├── README.md                         # Project documentation and reproduction instructions
├── requirements.txt                  # Pinned core Python environment (Python 3.13.7, pandas==3.1.1)
├── requirements-spark.txt            # Separate isolated PySpark environment (pyspark==4.1.4, pandas==2.2.3)
├── fdic_download_starter.py          # Part 1: FDIC BankFind API downloader & completeness checker
├── benchmark_storage.py              # Part 1: 5-round AB/BA CSV vs. Snappy Parquet benchmark (2023 cohort; supports --root)
├── src/
│   ├── part2_panel.py                # Part 2: quality audit, clean bank-quarter panel, 4Q failure labels
│   ├── part3_processing.py           # Part 3: pandas vs. Polars paired benchmark & Spark parity/plans
│   ├── part4_features.py             # Part 4: 1Q-lagged 14 CAMELS ratios, splits & train-only preprocessing
│   ├── part5_model.py                # Part 5: 3-fold expanding purged CV & shared helpers (legacy final test disabled)
│   ├── part5_policy_v3.py            # Part 5: authoritative V3 bank-quarter threshold calibration & locked test runner
│   └── part5_spark_demo.py           # Part 5: isolated development-only local Spark MLlib demonstration
├── tests/
│   ├── test_part2_panel.py           # Part 2 synthetic unittest suite (6 test cases)
│   ├── test_part3_processing.py      # Part 3 synthetic unittest suite (3 test cases)
│   ├── test_part4_features.py        # Part 4 synthetic unittest suite (6 test cases)
│   ├── test_part5_model.py           # Part 5 model/fold synthetic pytest suite (9 test cases)
│   └── test_part5_policy_v3.py       # Part 5 V3 policy & lock-guard synthetic pytest suite (11 test cases)
├── reports/
│   ├── benchmark_environment.json    # Authoritative Nov 11 evening benchmark system/software metadata
│   ├── benchmark_raw_measurements.csv# Authoritative 5-round AB/BA raw latency & file-size runs
│   └── benchmark_summary.csv         # Authoritative Part 1 storage benchmark medians
└── results/
    ├── part1/                        # Relocated & derived Part 1 aggregate acquisition & API reconciliation evidence
    ├── part2/                        # Part 2 aggregate quality, label, exit-code, event & redacted summary files
    ├── part3/                        # Part 3 benchmark metrics, 3-engine parity summary & sanitized execution plans
    ├── part4/                        # Part 4 split audit, 14-feature schema, missingness & preprocessor medians
    └── part5/                        # Part 5 CV audits, authoritative V3 calibration/test metrics, lock & Spark demo
```

---

## 3. Environment Requirements

### Recorded Research Environment Versions
Saved JSON metadata (`reports/benchmark_environment.json`, `results/part3/environment.json`, `results/part4/pipeline_audit.json`, `results/part5/development_selection_v3.json`) and local workspace virtual environment metadata establish the following exact versions used to produce the archived results:

- **Operating System:** Windows 11 (`win32`, `Windows-11-10.0.26300-SP0`, 4 logical cores reported in Part 1). Cross-platform Linux/macOS Python execution is expected to be compatible for Parts 1–5 pandas/Polars/scikit-learn scripts, but only Windows was tested.
- **Python Interpreter:** `3.13.7`
- **Core Environment (`requirements.txt`):** `pandas==3.1.1`, `numpy==2.5.4`, `pyarrow==26.0.1`, `polars==2.1.0`, `scikit-learn==1.9.1`, `joblib==1.6.0`, `psutil==7.2.2`, `matplotlib==3.11.2`, `pytest==9.2.0`, `requests==2.34.2`.
- **Separate Spark Environment (`requirements-spark.txt`):** `pyspark==4.1.4`, `py4j==0.10.9.9`, `pandas==2.2.3`, `pyarrow==26.0.1`, `numpy==2.5.4`.
  - **Why separate:** The workspace Spark environment pins `pandas==2.2.3` rather than `3.1.1` for PySpark compatibility.
  - **Java/JDK Prerequisite:** Running Spark stages requires a PySpark-compatible Java Development Kit (`java` on `PATH` and `JAVA_HOME` configured). Note that the exact host JDK version from the original run was **not** recorded in `results/part3/environment.json` or `results/part5/spark_mllib_demo.json`.

### Setup Instructions

Run commands from the repository root (`src/` uses PEP 420 implicit namespace imports, so invoke modules via `python -m src.<module>` from the root directory):

```bash
# 1. Core pipeline & test environment
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS:        source .venv/bin/activate
python -m pip install -r requirements.txt

# 2. Optional separate environment for PySpark stages (requires JDK on PATH / JAVA_HOME)
python -m venv .venv_spark
# Windows PowerShell: .\.venv_spark\Scripts\Activate.ps1
# Linux/macOS:        source .venv_spark/bin/activate
python -m pip install -r requirements-spark.txt
```

---

## 4. Data Acquisition

Raw and processed FDIC datasets are **not** committed to this repository. Data are obtained directly from the public **FDIC BankFind API** (`https://api.fdic.gov/banks`), which requires no account or API key:

- **Script Entry Point:** `fdic_download_starter.py`
- **Verified CLI Arguments:**
  - `--start` (default `"1984-03-31"`, `YYYY-MM-DD`)
  - `--end` (default `"2026-06-30"`, `YYYY-MM-DD`, or `"latest"`)
  - `--out` (default `"data_raw/fdic"`)
- **Outputs Written Under `<out>`:**
  - `financials/REPDTE=YYYYMMDD.parquet` (`170` files, Snappy-compressed via PyArrow, `33` fields total, ~`236.78 MB`)
  - `failures.parquet` (`4,117 × 9`, ~`201.66 KB`)
  - `institutions.parquet` (`27,835 × 9`, ~`836.44 KB`)
  - `manifest_financials.json` (per-quarter row vs. API-total verification; `0` incomplete quarters and `0` duplicate `(CERT, REPDTE)` keys expected)
- **Workload & Network Expectation:** Full 170-quarter download (~1.68M rows) runs politely with a `1.0 s` pause between API pages (`PAGE_SIZE = 10,000`) and takes roughly 1 hour depending on network throughput; resumable per-quarter skipping is built in.

> **Archived Part 1 Audit Note:** While `manifest_financials.json` is generated directly by `fdic_download_starter.py`, no standalone `.py` generator script exists in the workspace for `results/part1/acquisition_summary.json` or the original `reports/part1_evidence_summary.json` (from which `results/part1/acquisition_reconciliation_derived.json` was extracted). Both files are preserved as verified aggregate audit records from the original experiment.

---

## 5. Complete Reproduction Workflow

To preserve the archived experimental record under `reports/` and `results/`, point **fresh reproduction runs** at an external working root `<WORK_ROOT>` (e.g., `../fdic_repro_work`). Replace `<WORK_ROOT>` below with a clean, empty directory outside the committed repository evidence.

> **Execution Order Constraint (`part3_processing.py`):** Always run `--benchmark` **before** `--spark`. Executing `--benchmark` writes `aggregation_result.csv` and resets `"spark": "not_yet_run"` inside `engine_parity.json`; `--spark` depends on `aggregation_result.csv` as its reference table.

```bash
# Step 1: Acquire FDIC BankFind Financials (170Q), Failures, and Institutions (~1 hour full run)
python fdic_download_starter.py --start 1984-03-31 --end 2026-06-30 --out <WORK_ROOT>/data_raw/fdic

# Step 2 (Optional): Storage Benchmark (CSV vs. Snappy Parquet, 2023 cohort)
# Reads <WORK_ROOT>/data_raw/fdic/financials and writes under <WORK_ROOT>/reports/
# (Omitting --root uses the repository root; requires --root to keep archived reports/ untouched)
python benchmark_storage.py --root <WORK_ROOT>

# Step 3: Part 2 — Quality Audit, Panel Cleaning, and 4Q Failure Labels
python -m src.part2_panel --raw <WORK_ROOT>/data_raw/fdic --out <WORK_ROOT>/data_processed/part2 --results <WORK_ROOT>/results/part2

# Step 4a: Part 3 — Paired pandas vs. Polars Benchmark (MUST run before --spark)
python -m src.part3_processing --benchmark --panel <WORK_ROOT>/data_processed/part2/panel --labels <WORK_ROOT>/data_processed/part2/labels --results <WORK_ROOT>/results/part3

# Step 4b: Part 3 — Spark Parity Check & Plans (run in the separate Spark environment with JDK configured)
python -m src.part3_processing --spark --panel <WORK_ROOT>/data_processed/part2/panel --labels <WORK_ROOT>/data_processed/part2/labels --results <WORK_ROOT>/results/part3

# Step 5: Part 4 — 1Q-Lagged 14 CAMELS Features, Chronological Splits & Train-Only Preprocessing
python -m src.part4_features --root <WORK_ROOT>

# Step 6: Part 5 — 3-Fold Expanding Purged Development Cross-Validation
# (Supplies cv_fold_audit.csv, development_selection.json, and OOF predictions needed by Step 7)
python -m src.part5_model --phase development --root <WORK_ROOT>

# Step 7: Part 5 — Authoritative V3 Bank-Quarter Cost-Threshold Recalibration (Development Only)
python -m src.part5_policy_v3 --phase recalibrate --root <WORK_ROOT>

# Step 8 (Optional): Isolated Spark MLlib Development Demonstration (run in Spark environment)
python -m src.part5_spark_demo --root <WORK_ROOT>

# Step 9 (CONTROLLED / CONDITIONAL ONE-SHOT STEP — Fresh <WORK_ROOT> ONLY):
# Evaluate held-out 2009Q1-2025Q2 test data strictly once AFTER Steps 6 and 7 succeed on <WORK_ROOT>:
# python -m src.part5_policy_v3 --phase final-test --root <WORK_ROOT> --confirm-final-test
```

---

## 6. Final-Test Lock and Reproduction Safety

- **Original Archived Experiment:**
  - The study's model (`hist_gradient_boosting`), V3 bank-quarter decision threshold (`0.015019594386141859`), and held-out test metrics were frozen after a single authorized evaluation logged in `results/part5/final_test_lock.json` (`2026-11-24T16:53:04.676931+00:00`).
  - `results/part5/final_test_lock.json` and `results/part5/development_selection_v3.json` maintain a SHA-256 verification chain linking original CV selection (`714dee63...`) and fold audit (`a56e3ed9...`) -> V3 selection (`85a39a9e...`) -> final lock (`dd30ccd8...`).
  - **Never delete, edit, or reset `results/part5/final_test_lock.json`** (or overwrite `results/part5/*_v3.*`) in this repository. Running `src.part5_model --phase final-test` raises `PermissionError` permanently; running `src.part5_policy_v3` against the repository root raises `FileExistsError` by design to protect archived evidence.
- **Fresh Independent Reproduction:**
  - All pipeline stages (`fdic_download_starter.py`, `benchmark_storage.py`, `part2_panel.py`, `part3_processing.py`, `part4_features.py`, `part5_model.py`, `part5_policy_v3.py`, and `part5_spark_demo.py`) accept explicit `--out`/`--raw`/`--panel`/`--labels`/`--results` or `--root` arguments, enabling a complete isolated reproduction inside an external `<WORK_ROOT>` without modifying committed repository files.
  - Note: For `benchmark_storage.py`, `--root <WORK_ROOT>` was added in the public release candidate so external reproduction does not overwrite archived `reports/benchmark_*` files; omitting `--root` preserves the default repository-root behavior.
  - Archived original aggregate files under `reports/` and `results/` are provided strictly for inspection and audit verification; a fresh reproduction writes its own `development_selection_v3.json` and conditional one-shot `final_test_lock.json` under `<WORK_ROOT>/results/part5/`.

---

## 7. Key Verified Results (Parts 1–5)

- **Storage Format (Q1, Nov 11 Finalized Benchmark, 2023 Cohort `18,798 × 33`):** Snappy Parquet reduced size by `37.2%` (`2.8138 MB` vs. `4.4828 MB` CSV) with faster median write (`0.0744 s` vs. `0.3582 s`, `4.82x`), full read (`0.0195 s` vs. `0.0579 s`, `2.97x`), and 5-column read (`0.0045 s` vs. `0.0200 s`, `4.43x`).
- **Processing Benchmark & Parity (Q1, `1,599,071` labelled rows -> `2,373` year-state groups):** Paired 3-round medians showed Polars (`0.4983 s`, `502.61 MiB` peak Python RSS) vs. pandas (`5.8044 s`, `692.60 MiB` peak Python RSS). Aggregate equivalence passed across pandas, Polars (`0.0` max abs diff), and Spark (`9.97466e-17` max `positive_label_rate` diff).
- **Data Quality & Study-Period Events (Q2):** All `1,682,616` panel rows (`170` quarters) retained with `0` duplicate `(CERT, REPDTE)` keys (`33` inter-quarter bank reporting gaps audited). From `3,259` study-period failure/assistance rows (`>= 1984-03-31`), `35` multi-event-date institutions reconcile cleanly (`31` ASSISTANCE+FAILURE, `1` multiple ASSISTANCE-only, `3` conflicting FAILURE dates); full 1934–2026 history (`44/37/4/3`) is separately documented in `results/part2/multi_event_reconciliation_appendix_audit.json`.
- **Feature Pipeline (Q2):** `14` CAMELS ratios lagged by `1` quarter; strict train-only `SimpleImputer(median)` + `RobustScaler()`. Development set: `1,155,374` rows (`8,623` positives); purge gap (`2008`): `33,070` rows; held-out test (`2009Q1–2025Q2`): `387,518` rows (`1,779` positives, `501` distinct failed banks).
- **Development Selection & V3 Cost Threshold (Q3):** Across `3` purged expanding folds (`724,466` OOF rows, `1,791` positives), HistGradientBoosting achieved higher mean fold AP (`0.16083`) than Logistic Regression (`0.09625`). V3 bank-quarter calibration (using median historical `1984–2007` failure `COST = $8,639.222k` from `2,237` events and `1%` false-alarm cost ratio `$86.39222k`) froze threshold **`0.015019594386141859`** (OOF `TP=1,521`, `FP=15,007`, `FN=270`, `TN=707,668`). *(Legacy `development_selection.json` threshold `0.01597` was superseded by V3 and is retained only as provenance.)*
- **Locked Held-Out Evaluation (`2009Q1–2025Q2`, One-Shot V3 Result):** **`AP = 0.47485`**, **`ROC-AUC = 0.97975`**, **`Recall = 0.91287`**, **`Precision = 0.15984`**, **`F1 = 0.27205`**, **`TN = 377,203 / FP = 8,536 / FN = 155 / TP = 1,624`**, 1% proxy cost `$2,076,523.40k` (`$5,358.52k` per `1,000` bank-quarters). Separate development-only Spark MLlib demo (`206,850` train / `71,443` val): `areaUnderPR = 0.03022`, `areaUnderROC = 0.88324`.

---

## 8. Tests and Verification

All unit test suites in `tests/` use **synthetic in-memory or temporary-directory fixtures only**; none read raw/processed FDIC Parquets, held-out test data, or `final_test_lock.json`.

- **Test Definitions in Repository:** `35` test cases across `5` files (`test_part2_panel.py`: 6, `test_part3_processing.py`: 3, `test_part4_features.py`: 6, `test_part5_model.py`: 9, `test_part5_policy_v3.py`: 11).
- **Phase 3 Verified Execution:** Running inside the core environment (`Python 3.13.7`, `pytest 9.2.0`) with bytecode/cache suppression to keep the staging tree clean:
  ```bash
  python -B -m pytest tests -v -p no:cacheprovider
  ```
  Observed result: **`35 passed in 32.89s` (0 failed, 0 skipped)**, including `test_polars_same_logic`.
- **Important Boundary:** Passing these 35 synthetic unit tests verifies core transformation, leakage-guard, fold-purging, and lock invariants, **not** full end-to-end execution against live FDIC API downloads or the complete 1.68M-row panel.

---

## 9. Limitations and Reproducibility Boundaries

1. **Period-End Date vs. Filing Release Vintage:** `REPDTE` represents quarter-end reporting date, not historical Call Report publication timestamps; the one-quarter lag mitigates look-ahead bias without proving real-time filing availability.
2. **Retrospective Cost Proxy:** FDIC `COST` reflects retrospective resolution-loss estimates (not inflation-adjusted observed intervention savings), and false-alarm supervisory costs (`0.5%`, `1%`, `5%`) are explicitly assumed scenarios rather than observed costs. Bank-quarter proxy costs charge per prediction quarter rather than per unique bank event.
3. **Hardware & Timing Variance:** Exact millisecond timings and RSS values in `reports/benchmark_*` and `results/part3/benchmark_*` depend on host hardware, storage I/O, OS caching, and software builds.
4. **Unscripted Archived Part 1 Audit JSONs:** `results/part1/acquisition_summary.json` and the source of `results/part1/acquisition_reconciliation_derived.json` were captured during Part 1 verification and lack standalone generator scripts in the codebase.
5. **Unpublished OOF Hash Target:** `results/part5/development_selection_v3.json` records `source_oof_sha256` (`13a6d6da...`) for row-level OOF predictions (`chosen_model_cv_predictions.parquet`), which remain excluded from GitHub to avoid publishing bank-level `CERT` records; that hash cannot be independently recomputed from the public aggregate evidence alone unless Part 5 development is regenerated locally.
6. **Local Spark Scope:** Spark Part 3 parity and Part 5 MLlib (`local[2]`, `AdaptiveSparkPlan isFinalPlan=false` captured plan) illustrate distributed API behavior on a single machine and do **not** measure multi-node cluster speedup.
7. **Report & Appendix Path Alignment:** Archived reports written against the original workspace reference `data_raw/fdic/{acquisition_summary,manifest_financials}.json` (mapped here to `results/part1/`), `reports/part1_evidence_summary.json` (extracted to `results/part1/acquisition_reconciliation_derived.json`), unredacted `results/part2/part2_summary.json` (sanitized here with `ambiguous_failure_certs` redacted), `results/part3/*_plan.txt` (path-redacted with `<PROJECT_ROOT>`), and excluded bank-level/row-level outputs (`results/part2/reporting_gaps.csv`, `results/part3/{aggregation_result,spark_aggregation_result}.csv`). See `EVIDENCE_INDEX.md` for the full mapping.

---

## 10. References and AI Assistance

- **Data Source & Schema Documentation:** Federal Deposit Insurance Corporation (FDIC) BankFind Suite API (`https://api.fdic.gov/banks/docs/`) and RIS View Field Properties (`https://api.fdic.gov/banks/docs/risview_properties.yaml`).
- **Course Starter Code & Software:** Course-provided starter script (`fdic_download_starter.py`) extended across Parts 1–5 using Python, `pandas`, `numpy`, `pyarrow`, `polars`, `scikit-learn`, `joblib`, `psutil`, `matplotlib`, `requests`, `pytest`, and Apache `pyspark`.
- **AI Assistance Attribution:** **ChatGPT** and the **Google Antigravity coding agent** assisted with pipeline implementation structuring, unit-test authoring, evidence-audit cross-checking, sanitization checks, and preparation of this documentation/release package under human direction and verification.
