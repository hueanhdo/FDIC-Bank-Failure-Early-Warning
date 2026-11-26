"""FDIC Part 3: identical bank-year/state aggregation in pandas, Polars and Spark."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

START, END = "19840331", "20250630"
KEYS = ["CERT", "REPDTE"]
FIN_COLS = KEYS + ["STNAME", "ASSET", "DEP"]
LAB_COLS = KEYS + ["y_fail_next4q"]
SORT_KEYS = ["year", "state"]
COUNT_COLS = ["bank_quarters", "distinct_banks", "positive_labels", "deposit_reported_rows"]
FLOAT_COLS = ["total_assets_usd000", "reported_deposits_usd000", "positive_label_rate"]
OUTPUT_COLS = SORT_KEYS + COUNT_COLS + FLOAT_COLS


def check_inputs(panel_dir: Path, labels_dir: Path) -> None:
    """Require the completed Part 2 outputs; do not re-download or rebuild them."""
    for folder in (panel_dir, labels_dir):
        if not folder.is_dir() or not list(folder.glob("*.parquet")):
            raise FileNotFoundError(f"Missing Part 2 Parquet files in: {folder}")
    panel_names = {p.name for p in panel_dir.glob("*.parquet")}
    label_names = {p.name for p in labels_dir.glob("*.parquet")}
    if panel_names != label_names or len(panel_names) != 170:
        raise ValueError("Part 2 inputs must have matching quarter files (170 each).")


def aggregate_pandas(fin: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """Eager baseline: filter, one-to-one label join, then state/year aggregation."""
    fin = fin[FIN_COLS]
    lab = lab[LAB_COLS]
    fin = fin.loc[fin.REPDTE.between(START, END) & fin.ASSET.gt(0)].copy()
    lab = lab.loc[lab.REPDTE.between(START, END)].copy()
    joined = fin.merge(lab, how="inner", on=KEYS, validate="one_to_one")
    joined = joined.loc[joined.y_fail_next4q.notna()].copy()
    joined["y_fail_next4q"] = joined["y_fail_next4q"].astype("int64")
    joined["year"] = joined.REPDTE.str[:4].astype("int32")
    joined["state"] = joined.STNAME.fillna("UNKNOWN").astype("str")
    grouped = (joined.groupby(SORT_KEYS, sort=True, dropna=False)
               .agg(bank_quarters=("CERT", "size"),
                    distinct_banks=("CERT", "nunique"),
                    positive_labels=("y_fail_next4q", "sum"),
                    deposit_reported_rows=("DEP", "count"),
                    total_assets_usd000=("ASSET", "sum"),
                    reported_deposits_usd000=("DEP", "sum"))
               .reset_index())
    grouped["positive_label_rate"] = grouped.positive_labels / grouped.bank_quarters
    return grouped[OUTPUT_COLS].sort_values(SORT_KEYS).reset_index(drop=True)


def run_pandas(panel_dir: Path, labels_dir: Path) -> pd.DataFrame:
    """Read only the necessary columns; include file read time in the benchmark."""
    fin = pd.read_parquet(panel_dir, columns=FIN_COLS)
    lab = pd.read_parquet(labels_dir, columns=LAB_COLS)
    return aggregate_pandas(fin, lab)


def polars_transform(fin, lab):
    """Use one shared LazyFrame transformation for tests and production."""
    import polars as pl

    fin = (fin.select(FIN_COLS)
           .filter(pl.col("REPDTE").is_between(pl.lit(START), pl.lit(END)) &
                   pl.col("ASSET").is_not_null() &
                   pl.col("ASSET").is_not_nan() & (pl.col("ASSET") > 0)))
    lab = lab.select(LAB_COLS).filter(pl.col("REPDTE").is_between(pl.lit(START), pl.lit(END)))
    return (fin.join(lab, on=KEYS, how="inner")
            .filter(pl.col("y_fail_next4q").is_not_null())
            .with_columns(
                pl.col("REPDTE").str.slice(0, 4).cast(pl.Int32).alias("year"),
                pl.col("STNAME").fill_null("UNKNOWN").alias("state"),
                pl.col("DEP").fill_nan(None),
            )
            .group_by(SORT_KEYS)
            .agg(pl.len().alias("bank_quarters"),
                 pl.col("CERT").n_unique().alias("distinct_banks"),
                 pl.col("y_fail_next4q").sum().alias("positive_labels"),
                 pl.col("DEP").is_not_null().sum().alias("deposit_reported_rows"),
                 pl.col("ASSET").sum().alias("total_assets_usd000"),
                 pl.col("DEP").sum().alias("reported_deposits_usd000"))
            .with_columns((pl.col("positive_labels") / pl.col("bank_quarters"))
                          .alias("positive_label_rate"))
            .select(OUTPUT_COLS).sort(SORT_KEYS))


def polars_query(panel_dir: Path, labels_dir: Path):
    """Build a lazy Parquet plan; data are read only on collect()."""
    import polars as pl

    fin = pl.scan_parquet((panel_dir.resolve() / "*.parquet").as_posix(), hive_partitioning=False)
    lab = pl.scan_parquet((labels_dir.resolve() / "*.parquet").as_posix(), hive_partitioning=False)
    return polars_transform(fin, lab)


def aggregate_polars(fin, lab):
    """Small in-memory test adapter that uses the EXACT production operations."""
    return polars_transform(fin, lab)


def run_polars(panel_dir: Path, labels_dir: Path) -> pd.DataFrame:
    """Collect once: only the aggregated output is converted to pandas."""
    return polars_query(panel_dir, labels_dir).collect().to_pandas()


def spark_query(spark, panel_dir: Path, labels_dir: Path):
    """Spark SQL equivalent; transformations remain lazy until an action."""
    from pyspark.sql import functions as F

    panel_files = [str(p.resolve()) for p in sorted(panel_dir.glob("*.parquet"))]
    labels_files = [str(p.resolve()) for p in sorted(labels_dir.glob("*.parquet"))]
    fin = (spark.read.parquet(*panel_files)
           .select(*FIN_COLS)
           .filter(F.col("REPDTE").between(START, END) &
                   F.col("ASSET").isNotNull() &
                   ~F.isnan("ASSET") & (F.col("ASSET") > 0)))
    lab = (spark.read.parquet(*labels_files)
           .select(*LAB_COLS).filter(F.col("REPDTE").between(START, END)))
    joined = (fin.join(lab, KEYS, "inner")
              .filter(F.col("y_fail_next4q").isNotNull())
              .withColumn("year", F.substring("REPDTE", 1, 4).cast("integer"))
              .withColumn("state", F.coalesce(F.col("STNAME"), F.lit("UNKNOWN")))
              .withColumn("DEP", F.when(F.isnan("DEP"), F.lit(None)).otherwise(F.col("DEP"))))
    grouped = (joined.groupBy(*SORT_KEYS)
               .agg(F.count("CERT").alias("bank_quarters"),
                    F.countDistinct("CERT").alias("distinct_banks"),
                    F.sum(F.col("y_fail_next4q").cast("long")).alias("positive_labels"),
                    F.count("DEP").alias("deposit_reported_rows"),
                    F.sum("ASSET").alias("total_assets_usd000"),
                    F.coalesce(F.sum("DEP"), F.lit(0.0)).alias("reported_deposits_usd000"))
               .withColumn("positive_label_rate", F.col("positive_labels") / F.col("bank_quarters")))
    return grouped.select(*OUTPUT_COLS).orderBy(*SORT_KEYS)


def compare_outputs(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    """Require identical group keys/counts and numerically equivalent aggregates."""
    ref = reference[OUTPUT_COLS].sort_values(SORT_KEYS).reset_index(drop=True)
    got = candidate[OUTPUT_COLS].sort_values(SORT_KEYS).reset_index(drop=True)
    if len(ref) != len(got) or ref[SORT_KEYS].astype("string").fillna("<NA>").to_numpy().tolist() != got[SORT_KEYS].astype("string").fillna("<NA>").to_numpy().tolist():
        raise AssertionError("Group keys differ across engines.")
    for name in COUNT_COLS:
        if not np.array_equal(ref[name].astype("int64"), got[name].astype("int64")):
            raise AssertionError(f"Count mismatch: {name}")
    max_differences = {}
    for name in FLOAT_COLS:
        a, b = ref[name].astype("float64").to_numpy(), got[name].astype("float64").to_numpy()
        if not np.allclose(a, b, rtol=1e-8, atol=1e-4, equal_nan=True):
            raise AssertionError(f"Numeric mismatch: {name}")
        max_differences[name] = float(np.max(np.abs(a - b), initial=0.0))
    return {"passed": True, "groups": len(ref),
            "labelled_bank_quarters": int(ref["bank_quarters"].sum()),
            "positive_labels": int(ref["positive_labels"].sum()),
            "max_absolute_differences": max_differences}


def sampled_memory_runner(operation) -> tuple[pd.DataFrame, float, float, float]:
    """Sample current Python process RSS (not JVM), every 10 ms during operation."""
    import psutil

    process = psutil.Process(os.getpid())
    initial = process.memory_info().rss
    maximum = [initial]
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            try:
                maximum[0] = max(maximum[0], process.memory_info().rss)
            except psutil.Error:
                break
            stop.wait(0.01)

    monitor = threading.Thread(target=watch, daemon=True)
    monitor.start()
    t0 = time.perf_counter()
    try:
        result = operation()
    finally:
        elapsed = time.perf_counter() - t0
        stop.set()
        monitor.join()
    maximum[0] = max(maximum[0], process.memory_info().rss)
    mib = 1024 ** 2
    return result, elapsed, maximum[0] / mib, (maximum[0] - initial) / mib


def worker(args) -> None:
    """Run exactly one timed trial in a fresh process; save tiny result outside timing."""
    if args.worker == "pandas":
        import pyarrow  # noqa: F401; pre-load I/O engine before the timer
        operation = lambda: run_pandas(args.panel, args.labels)
    else:
        import polars  # noqa: F401
        operation = lambda: run_polars(args.panel, args.labels)
    output, elapsed, peak, extra = sampled_memory_runner(operation)
    output.to_csv(args.output, index=False)
    print(json.dumps({"engine": args.worker, "seconds": elapsed,
                      "peak_python_rss_mib": peak, "incremental_python_rss_mib": extra,
                      "groups": len(output), "rows": int(output.bank_quarters.sum())}))


def benchmark(args) -> None:
    """Three interleaved pairs, no automatic retries; fail fast on any discrepancy."""
    check_inputs(args.panel, args.labels)
    args.results.mkdir(parents=True, exist_ok=True)
    import polars as pl
    import pyarrow

    plan = polars_query(args.panel, args.labels).explain(optimized=True)
    (args.results / "polars_optimized_plan.txt").write_text(plan, encoding="utf-8")
    records = []
    reference = None
    # AB / BA / AB reduces the risk of always favoring the warm file cache.
    for iteration, order in enumerate((("pandas", "polars"), ("polars", "pandas"), ("pandas", "polars")), 1):
        for engine in order:
            temporary = args.results / f"_trial_{engine}.csv"
            cmd = [sys.executable, "-m", "src.part3_processing", "--worker", engine,
                   "--panel", str(args.panel), "--labels", str(args.labels),
                   "--output", str(temporary)]
            try:
                run = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=True)
                metrics = json.loads(run.stdout.strip().splitlines()[-1])
                result = pd.read_csv(temporary)
                if reference is None:
                    reference = result.copy()
                    reference.to_csv(args.results / "aggregation_result.csv", index=False)
                evidence = compare_outputs(reference, result)
                records.append({"iteration": iteration, "engine": engine, **metrics})
                print(f"{engine} #{iteration}: {metrics['seconds']:.3f}s | peak RSS {metrics['peak_python_rss_mib']:.1f} MiB | parity PASS")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, AssertionError) as exc:
                err = getattr(exc, "stderr", "")
                raise RuntimeError(f"Benchmark failed for {engine} on trial {iteration}. {err or exc}") from exc
            finally:
                temporary.unlink(missing_ok=True)
    trials = pd.DataFrame(records)
    trials.to_csv(args.results / "benchmark_runs.csv", index=False)
    summary = (trials.groupby("engine", as_index=False)
               [["seconds", "peak_python_rss_mib", "incremental_python_rss_mib"]]
               .median().rename(columns={"seconds": "median_seconds",
                                        "peak_python_rss_mib": "median_peak_python_rss_mib",
                                        "incremental_python_rss_mib": "median_incremental_python_rss_mib"}))
    summary.to_csv(args.results / "benchmark_summary.csv", index=False)
    (args.results / "engine_parity.json").write_text(
        json.dumps({"pandas_vs_polars": evidence, "spark": "not_yet_run"}, indent=2), encoding="utf-8")
    (args.results / "environment.json").write_text(json.dumps({
        "python": sys.version.split()[0], "pandas": pd.__version__, "polars": pl.__version__,
        "pyarrow": pyarrow.__version__, "platform": sys.platform,
        "benchmark_design": "3 interleaved pairs, fresh subprocess, read+transform+aggregate; no child process startup in timer",
        "memory_measure": "sampled Python-process RSS at 10ms intervals; JVM memory excluded",
        "created_utc": datetime.now(timezone.utc).isoformat()
    }, indent=2), encoding="utf-8")
    print("Pandas/Polars benchmark PASS; CSV/JSON and optimized Polars plan saved.")


def spark_demo(args) -> None:
    """Run Spark local once; save both plan views and compare against benchmark."""
    check_inputs(args.panel, args.labels)
    baseline = args.results / "aggregation_result.csv"
    if not baseline.is_file():
        raise FileNotFoundError("Run --benchmark before --spark to establish a reference result.")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession

    spark = (SparkSession.builder.master("local[2]").appName("FDIC-Part3-Local")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        query = spark_query(spark, args.panel, args.labels)
        for mode in ("formatted", "extended"):
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                query.explain(mode=mode)
            (args.results / f"spark_{mode}_plan.txt").write_text(capture.getvalue(), encoding="utf-8")
        # Only the small grouped result is brought back to Python, not the full 1.6m-row panel.
        output = query.toPandas()
        check = compare_outputs(pd.read_csv(baseline), output)
        output.to_csv(args.results / "spark_aggregation_result.csv", index=False)
        parity_file = args.results / "engine_parity.json"
        parity = json.loads(parity_file.read_text(encoding="utf-8"))
        parity["spark"] = check
        parity_file.write_text(json.dumps(parity, indent=2), encoding="utf-8")
        print(f"Spark parity PASS: {check['groups']} year/state groups; plans saved.")
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--benchmark", action="store_true", help="Benchmark pandas/Polars: 3 paired trials each")
    mode.add_argument("--spark", action="store_true", help="Spark local parity and logical/physical plans")
    mode.add_argument("--worker", choices=["pandas", "polars"], help=argparse.SUPPRESS)
    parser.add_argument("--panel", type=Path, default=Path("data_processed/part2/panel"))
    parser.add_argument("--labels", type=Path, default=Path("data_processed/part2/labels"))
    parser.add_argument("--results", type=Path, default=Path("results/part3"))
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.output is None:
            parser.error("--worker requires --output")
        worker(args)
    elif args.benchmark:
        benchmark(args)
    else:
        spark_demo(args)


if __name__ == "__main__":
    main()
