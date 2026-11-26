"""
BFT65B Final Report - Storage Layout & Empirical Benchmark Script.
Compares raw CSV against Snappy-compressed Parquet on identical FDIC Call Report data.

METHODOLOGICAL NOTE:
This is a single-machine local I/O benchmark on a consolidated one-year sample
(2023 Call Report cohort: 18,798 observations x 33 columns). It isolates format
and compression efficiency (CSV vs. Parquet Snappy) under identical records.
It is NOT a direct benchmark of Hive partition pruning or the full 170-quarter panel.

METHODOLOGICAL IMPROVEMENTS:
1. Balanced Alternating Execution Order: Alternates execution between Parquet-first
   and CSV-first across rounds (AB/BA design) to cancel out OS cache warming bias.
2. Data-Equivalence Checks: Rigorously verifies that read outputs from CSV and Parquet
   are identical in shape, column names, null distributions, and numerical values
   within reasonable floating-point tolerance (rtol=1e-5, atol=1e-5).
3. Environment Logging: Exports exact hardware and software specifications
   to reports/benchmark_environment.json.

Environment:
- Windows 11 (4 cores)
- Python 3.13.7, pandas 3.1.1, pyarrow 26.0.1, matplotlib 3.11.2

Usage:
    python benchmark_storage.py
    python benchmark_storage.py --root <WORK_ROOT>
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

# -----------------------------------------------------------------------------
# 1. Benchmark Configuration
# -----------------------------------------------------------------------------
SAMPLE_QUARTERS = ["20230331", "20230630", "20230930", "20231231"]  # All 4 quarters of 2023
SELECTED_COLUMNS = ["CERT", "REPDTE", "ASSET", "DEP", "ROA"]          # 5 key CAMELS columns
N_ROUNDS = 5                                                           # 5 repeated trials

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data_raw" / "fdic"
REPORTS_DIR = ROOT_DIR / "reports"
TEMP_DIR = REPORTS_DIR / "temp_benchmark_data"


def load_benchmark_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load identical records and columns across all 4 quarters of 2023."""
    dfs = []
    for q in SAMPLE_QUARTERS:
        p = data_dir / "financials" / f"REPDTE={q}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing quarter parquet: {p}. Run acquisition first.")
        dfs.append(pd.read_parquet(p))
    df = pd.concat(dfs, ignore_index=True)
    return df


def verify_data_equivalence(df_csv: pd.DataFrame, df_pq: pd.DataFrame, context: str = "Full Table") -> None:
    """Verify that CSV and Parquet outputs represent identical underlying data."""
    # 1. Dimensions
    if df_csv.shape != df_pq.shape:
        raise AssertionError(f"[{context}] Shape mismatch: CSV {df_csv.shape} vs Parquet {df_pq.shape}")
    
    # 2. Column set equivalence and alignment
    if set(df_csv.columns) != set(df_pq.columns):
        raise AssertionError(f"[{context}] Column set mismatch: CSV {set(df_csv.columns)} vs Parquet {set(df_pq.columns)}")
    
    # Align column order for direct comparison (read_csv usecols retains CSV header order)
    cols = list(df_pq.columns)
    df_csv = df_csv[cols]
    
    # 3. Key column equivalence
    if "CERT" in df_csv.columns and "REPDTE" in df_csv.columns:
        if not (df_csv["CERT"].astype(int) == df_pq["CERT"].astype(int)).all():
            raise AssertionError(f"[{context}] CERT key values differ between CSV and Parquet")
        if not (df_csv["REPDTE"].astype(str) == df_pq["REPDTE"].astype(str)).all():
            raise AssertionError(f"[{context}] REPDTE date values differ between CSV and Parquet")

    # 4. Values and null handling across columns
    for col in df_csv.columns:
        s_csv = df_csv[col]
        s_pq = df_pq[col]
        
        # Numerical comparison
        if pd.api.types.is_numeric_dtype(s_pq) and pd.api.types.is_numeric_dtype(s_csv):
            # Check null alignment for numbers
            if not (s_csv.isna() == s_pq.isna()).all():
                raise AssertionError(f"[{context}] Numeric null position mismatch in column {col}")
            valid_mask = ~s_csv.isna()
            if valid_mask.any():
                v_csv = s_csv[valid_mask].astype(float).to_numpy()
                v_pq = s_pq[valid_mask].astype(float).to_numpy()
                if not np.allclose(v_csv, v_pq, rtol=1e-5, atol=1e-5, equal_nan=True):
                    max_diff = np.max(np.abs(v_csv - v_pq))
                    raise AssertionError(f"[{context}] Numerical divergence in column {col}: max diff = {max_diff}")
        else:
            # String / object comparison (handling CSV standard parsing where empty strings become NaN)
            str_csv = s_csv.fillna("").astype(str)
            str_pq = s_pq.fillna("").astype(str)
            if not (str_csv == str_pq).all():
                diff_count = (str_csv != str_pq).sum()
                raise AssertionError(f"[{context}] Text value divergence in column {col}: {diff_count} mismatches")


def run_benchmark(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base_root = ROOT_DIR if root is None else Path(root).resolve()
    data_dir = base_root / "data_raw" / "fdic"
    reports_dir = base_root / "reports"
    temp_dir = reports_dir / "temp_benchmark_data"
    reports_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(reports_dir / ".matplotlib_cache"))
    import matplotlib

    print("=================================================================")
    print("BFT65B PART 1: STORAGE BENCHMARK (CSV VS PARQUET SNAPPY)")
    print("=================================================================")
    print("SCOPE NOTE: Single-machine consolidated one-year sample (2023).")
    print("Not a direct benchmark of partition pruning or the full 170-quarter panel.\n")
    
    df = load_benchmark_dataset(data_dir=data_dir)
    n_rows, n_cols = df.shape
    print(f"Sample Period     : 2023Q1 -> 2023Q4 ({len(SAMPLE_QUARTERS)} quarters)")
    print(f"Dataset Dimensions: {n_rows:,} rows x {n_cols} columns")
    print(f"Selected Columns  : {SELECTED_COLUMNS}")
    print(f"Number of Repeats : {N_ROUNDS} rounds (Alternating AB/BA order)")
    print("Methodology       : Balanced execution order + data equivalence verification\n")

    csv_path = temp_dir / "benchmark_sample.csv"
    pq_path = temp_dir / "benchmark_sample.parquet"

    raw_runs = []

    for round_idx in range(1, N_ROUNDS + 1):
        # Balanced alternating design:
        # Odd rounds (1, 3, 5): Parquet first, then CSV (P -> C)
        # Even rounds (2, 4)  : CSV first, then Parquet (C -> P)
        parity = "Parquet -> CSV" if round_idx % 2 == 1 else "CSV -> Parquet"
        print(f"--- Running Round {round_idx}/{N_ROUNDS} (Order: {parity}) ---")
        
        # -------------------------------------------------------------
        # 1. Write Benchmark
        # -------------------------------------------------------------
        if round_idx % 2 == 1:
            # P then C
            t0 = time.perf_counter()
            df.to_parquet(pq_path, index=False, engine="pyarrow", compression="snappy")
            t_write_pq = time.perf_counter() - t0

            t0 = time.perf_counter()
            df.to_csv(csv_path, index=False)
            t_write_csv = time.perf_counter() - t0
        else:
            # C then P
            t0 = time.perf_counter()
            df.to_csv(csv_path, index=False)
            t_write_csv = time.perf_counter() - t0

            t0 = time.perf_counter()
            df.to_parquet(pq_path, index=False, engine="pyarrow", compression="snappy")
            t_write_pq = time.perf_counter() - t0

        # -------------------------------------------------------------
        # 2. Measure File Sizes (MB)
        # -------------------------------------------------------------
        size_pq_mb = pq_path.stat().st_size / (1024 * 1024)
        size_csv_mb = csv_path.stat().st_size / (1024 * 1024)

        # -------------------------------------------------------------
        # 3. Full Read Benchmark
        # -------------------------------------------------------------
        if round_idx % 2 == 1:
            t0 = time.perf_counter()
            df_read_pq = pd.read_parquet(pq_path)
            t_read_full_pq = time.perf_counter() - t0

            t0 = time.perf_counter()
            df_read_csv = pd.read_csv(csv_path)
            t_read_full_csv = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            df_read_csv = pd.read_csv(csv_path)
            t_read_full_csv = time.perf_counter() - t0

            t0 = time.perf_counter()
            df_read_pq = pd.read_parquet(pq_path)
            t_read_full_pq = time.perf_counter() - t0

        # Verify data equivalence on full read
        verify_data_equivalence(df_read_csv, df_read_pq, context=f"Round {round_idx} Full Table")

        # -------------------------------------------------------------
        # 4. Selected-Column Read Benchmark (Projection)
        # -------------------------------------------------------------
        if round_idx % 2 == 1:
            t0 = time.perf_counter()
            df_sub_pq = pd.read_parquet(pq_path, columns=SELECTED_COLUMNS)
            t_read_subset_pq = time.perf_counter() - t0

            t0 = time.perf_counter()
            df_sub_csv = pd.read_csv(csv_path, usecols=SELECTED_COLUMNS)
            t_read_subset_csv = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            df_sub_csv = pd.read_csv(csv_path, usecols=SELECTED_COLUMNS)
            t_read_subset_csv = time.perf_counter() - t0

            t0 = time.perf_counter()
            df_sub_pq = pd.read_parquet(pq_path, columns=SELECTED_COLUMNS)
            t_read_subset_pq = time.perf_counter() - t0

        # Verify data equivalence on projection read
        verify_data_equivalence(df_sub_csv, df_sub_pq, context=f"Round {round_idx} 5-Column Projection")

        raw_runs.append({
            "round": round_idx,
            "order": parity,
            "write_csv_s": t_write_csv,
            "write_parquet_s": t_write_pq,
            "read_full_csv_s": t_read_full_csv,
            "read_full_parquet_s": t_read_full_pq,
            "read_subset_csv_s": t_read_subset_csv,
            "read_subset_parquet_s": t_read_subset_pq,
            "size_csv_mb": size_csv_mb,
            "size_parquet_mb": size_pq_mb
        })

    df_runs = pd.DataFrame(raw_runs)

    # Calculate summary metrics (Median, Mean, Std)
    metrics = [
        ("File Size (MB)", "size_csv_mb", "size_parquet_mb"),
        ("Write Time (s)", "write_csv_s", "write_parquet_s"),
        ("Full Read Time (s)", "read_full_csv_s", "read_full_parquet_s"),
        ("5-Column Read Time (s)", "read_subset_csv_s", "read_subset_parquet_s"),
    ]

    summary_rows = []
    for label, csv_col, pq_col in metrics:
        csv_med = float(df_runs[csv_col].median())
        pq_med = float(df_runs[pq_col].median())
        
        csv_mean = float(df_runs[csv_col].mean())
        pq_mean = float(df_runs[pq_col].mean())
        
        csv_std = float(df_runs[csv_col].std())
        pq_std = float(df_runs[pq_col].std())
        
        if "Size" in label:
            pct_diff = ((pq_med - csv_med) / csv_med) * 100
            advantage = f"{abs(pct_diff):.1f}% space saved" if pct_diff < 0 else f"{pct_diff:.1f}% larger"
        else:
            speedup = csv_med / pq_med if pq_med > 0 else 0
            advantage = f"{speedup:.2f}x faster" if speedup >= 1 else f"{1/speedup:.2f}x slower"

        summary_rows.append({
            "Metric": label,
            "CSV (Median)": round(csv_med, 4),
            "CSV (Mean ± Std)": f"{csv_mean:.4f} ± {csv_std:.4f}",
            "Parquet Snappy (Median)": round(pq_med, 4),
            "Parquet Snappy (Mean ± Std)": f"{pq_mean:.4f} ± {pq_std:.4f}",
            "Empirical Advantage": advantage
        })

    df_summary = pd.DataFrame(summary_rows)

    env_info = {
        "benchmark_timestamp": datetime.now().isoformat(),
        "benchmark_scope": "Single-machine consolidated one-year sample (2023 Call Report cohort)",
        "disclaimer": "This benchmark measures single-machine format efficiency under identical rows/cols. It is NOT a direct benchmark of Hive partition pruning or the full 170-quarter panel.",
        "methodology": "Balanced alternating execution order (AB/BA design over 5 rounds) with data-equivalence checks",
        "sample": {
            "period": "2023Q1-2023Q4",
            "quarters": SAMPLE_QUARTERS,
            "rows": n_rows,
            "columns": n_cols,
            "selected_columns": SELECTED_COLUMNS
        },
        "system_environment": {
            "os": platform.platform(),
            "processor": platform.processor(),
            "logical_cores": os.cpu_count(),
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "pyarrow_version": pyarrow.__version__,
            "matplotlib_version": matplotlib.__version__
        },
        "rounds": N_ROUNDS,
        "data_equivalence_verified": True
    }

    # Clean up temp benchmark files
    shutil.rmtree(temp_dir, ignore_errors=True)

    return df_runs, df_summary, env_info


def generate_charts(df_summary: pd.DataFrame, df_runs: pd.DataFrame, reports_dir: Path = REPORTS_DIR) -> None:
    """Generate publication-ready academic figures."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(reports_dir / ".matplotlib_cache"))
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # -------------------------------------------------------------
    # Figure 1: File Size Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    csv_size = df_summary.loc[df_summary["Metric"] == "File Size (MB)", "CSV (Median)"].values[0]
    pq_size = df_summary.loc[df_summary["Metric"] == "File Size (MB)", "Parquet Snappy (Median)"].values[0]
    
    bars = ax.bar(["Raw CSV", "Parquet (Snappy)"], [csv_size, pq_size], color=["#e74c3c", "#2980b9"], width=0.55)
    ax.set_ylabel("Storage Size (MB)", fontsize=11, fontweight="bold")
    ax.set_title("Empirical Storage Size: 2023 FDIC Panel (18,798 Rows × 33 Cols)", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylim(0, max(csv_size, pq_size) * 1.25)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{h:.2f} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    diff_pct = (1 - pq_size / csv_size) * 100
    ax.annotate(f"Space Saved:\n{diff_pct:.1f}% Reduction",
                xy=(1, pq_size), xytext=(0.5, (csv_size + pq_size) / 2),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5),
                ha="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="#ecf0f1", ec="#bdc3c7"))

    plt.tight_layout()
    fig1_path = reports_dir / "fig1_storage_size_comparison.png"
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Read Latency Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    full_csv = df_summary.loc[df_summary["Metric"] == "Full Read Time (s)", "CSV (Median)"].values[0] * 1000
    full_pq = df_summary.loc[df_summary["Metric"] == "Full Read Time (s)", "Parquet Snappy (Median)"].values[0] * 1000
    
    sub_csv = df_summary.loc[df_summary["Metric"] == "5-Column Read Time (s)", "CSV (Median)"].values[0] * 1000
    sub_pq = df_summary.loc[df_summary["Metric"] == "5-Column Read Time (s)", "Parquet Snappy (Median)"].values[0] * 1000
    
    x = np.arange(2)
    width = 0.35
    
    rects1 = ax.bar(x - width/2, [full_csv, sub_csv], width, label="Raw CSV", color="#e74c3c")
    rects2 = ax.bar(x + width/2, [full_pq, sub_pq], width, label="Parquet (Snappy)", color="#2980b9")
    
    ax.set_ylabel("Read Latency (Milliseconds - Lower is Better)", fontsize=10, fontweight="bold")
    ax.set_title("Empirical Read Latency: Full Read vs 5-Column Projection", fontsize=11, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["Full Table Read (33 cols)", "5-Column Projection (CERT, REPDTE, etc.)"], fontsize=10)
    ax.legend(frameon=True)
    ax.set_ylim(0, max(full_csv, full_pq) * 1.25)
    
    for r in rects1:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + 1, f"{h:.1f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + 1, f"{h:.1f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig2_path = reports_dir / "fig2_read_latency_comparison.png"
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------
    # Figure 3: Combined Academic Publication Dashboard (2 Panels)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    
    # Left Panel: Storage Size
    bars1 = ax1.bar(["Raw CSV", "Parquet (Snappy)"], [csv_size, pq_size], color=["#e74c3c", "#2980b9"], width=0.5)
    ax1.set_ylabel("File Size (MB)", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Disk Storage Footprint (MB)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(csv_size, pq_size) * 1.3)
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 0.1, f"{h:.2f} MB", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.text(0.5, (csv_size + pq_size)/2, f"Δ = -{diff_pct:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="#d5f5e3", ec="#27ae60"))

    # Right Panel: Latency across Operations
    ops = ["Write Time", "Full Read", "5-Col Read"]
    w_csv = df_summary.loc[df_summary["Metric"] == "Write Time (s)", "CSV (Median)"].values[0] * 1000
    w_pq = df_summary.loc[df_summary["Metric"] == "Write Time (s)", "Parquet Snappy (Median)"].values[0] * 1000
    
    y_csv = [w_csv, full_csv, sub_csv]
    y_pq = [w_pq, full_pq, sub_pq]
    
    x_ops = np.arange(len(ops))
    w_bar = 0.35
    b_csv = ax2.bar(x_ops - w_bar/2, y_csv, w_bar, label="Raw CSV", color="#e74c3c")
    b_pq = ax2.bar(x_ops + w_bar/2, y_pq, w_bar, label="Parquet (Snappy)", color="#2980b9")
    
    ax2.set_ylabel("Latency (Milliseconds - Log Scale)", fontsize=10, fontweight="bold")
    ax2.set_yscale("log")
    ax2.set_title("(b) I/O Latency Benchmark (Median over 5 Runs)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x_ops)
    ax2.set_xticklabels(ops, fontsize=10)
    ax2.legend(frameon=True)

    for b in b_csv:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, h * 1.1, f"{h:.1f}ms", ha="center", va="bottom", fontsize=8)
    for b in b_pq:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, h * 1.1, f"{h:.1f}ms", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("Empirical Comparison of Storage Formats on FDIC Financials Panel (2023 Cohort, N = 18,798)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig3_path = reports_dir / "fig3_benchmark_dashboard.png"
    plt.savefig(fig3_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig3_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run single-machine CSV vs. Snappy Parquet storage benchmark.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional working root containing data_raw/fdic/financials and receiving reports/ (default: repository root)",
    )
    args = parser.parse_args(argv)
    base_root = ROOT_DIR if args.root is None else args.root.resolve()
    reports_dir = base_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df_runs, df_summary, env_info = run_benchmark(root=base_root)

    # Save CSV outputs in reports/
    raw_path = reports_dir / "benchmark_raw_measurements.csv"
    summary_path = reports_dir / "benchmark_summary.csv"
    env_path = reports_dir / "benchmark_environment.json"
    
    df_runs.to_csv(raw_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    env_path.write_text(json.dumps(env_info, indent=2))
    
    print("\n=== BENCHMARK SUMMARY (MEDIAN OF 5 BALANCED REPEATS) ===")
    print(df_summary.to_string(index=False))
    print(f"\nSaved raw measurements: {raw_path}")
    print(f"Saved summary table   : {summary_path}")
    print(f"Saved environment info: {env_path}")

    # Generate academic charts
    print("\nGenerating charts...")
    generate_charts(df_summary, df_runs, reports_dir=reports_dir)
    print("Benchmark complete successfully with data equivalence verified.")


if __name__ == "__main__":
    main()
