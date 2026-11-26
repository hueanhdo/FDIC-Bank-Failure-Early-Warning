"""Part 4: conservative point-in-time FDIC features and train-only preprocessing.

Inputs (unchanged): Part 2 panel and labels, one Parquet per reporting quarter.
This module deliberately never uses Failures/Institutions tables or event fields.
Prediction at the end of quarter q uses FINANCIALS FROM q-1; target is y(q, q+4Q].
Financials release/vintage timestamps are unavailable: the one-quarter lag is a
conservative proxy, NOT a guarantee of actual real-time data availability.

Part 5 must instantiate a FRESH make_preprocessor() inside each time-validation
fold's full classifier Pipeline; do not reuse the pre-fitted development artifact
for cross-validation or threshold selection.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

LABEL = "y_fail_next4q"
KEYS = ["CERT", "REPDTE"]
SOURCE_DATE = "SOURCE_REPDTE"
FIRST = "19840331"
TRAIN_END = "20071231"
GAP_START, GAP_END = "20080331", "20081231"
TEST_START, TEST_END = "20090331", "20250630"
MIN_UNIQUE_POSITIVE_TEST_BANKS = 50

# Allowlist extracted from the 33 FDIC fields acquired by fdic_download_starter.py.
# Never use a column from Failures, Institutions or any other outcome lookup.
RAW_FIELDS = [
    "ASSET", "EQ", "RBC1AAJ", "LNLSGR", "NCLNLS", "P9ASSET",
    "NAASSET", "NTLNLS", "LNATRES", "BRO", "DEP", "CHBAL", "LNRE",
    "ROA", "ROE", "NIMY", "NONIX", "INTINC", "EINTEXP", "NONII",
]
FORBIDDEN = {
    LABEL, "FAILDATE", "FAILYR", "RESTYPE", "FAILURE", "COST", "QBFASSET",
    "QBFDEP", "ACTIVE", "CHANGEC1", "ENDEFYMD", "failure_date",
    "end_date", "future_failure", "event_type", "EXIT_TYPE", "event_status",
    "label", "target", "competing_exit", "ambiguous_failure",
}

# Numerators and denominators are all amount fields in $1,000 except the three
# FDIC-reported profitability percentages and RBC1AAJ (already in percent).
FEATURE_DEFINITIONS = {
    "equity_to_assets": {"formula": "EQ / ASSET", "interpretation": "Capital adequacy (book equity ratio)"},
    "tier1_leverage_pct": {"formula": "RBC1AAJ (as reported, percent)", "interpretation": "Reported regulatory capital proxy"},
    "noncurrent_loans_to_loans": {"formula": "NCLNLS / LNLSGR", "interpretation": "Noncurrent gross loans share"},
    "pastdue90_nonaccrual_assets_to_assets": {"formula": "(P9ASSET + NAASSET) / ASSET", "interpretation": "Serious problem assets over assets; NOT solely an NPL ratio"},
    "net_chargeoffs_to_loans": {"formula": "NTLNLS / LNLSGR", "interpretation": "Net charge-offs / gross loans; original flow conventions retained"},
    "loan_loss_allowance_to_loans": {"formula": "LNATRES / LNLSGR", "interpretation": "Loss allowance relative to loans, NOT allowance/NPL coverage"},
    "real_estate_loans_to_loans": {"formula": "LNRE / LNLSGR", "interpretation": "Real-estate concentration"},
    "efficiency_proxy": {"formula": "NONIX / (INTINC - EINTEXP + NONII)", "interpretation": "Expense intensity; imperfect management-quality proxy"},
    "roa_pct": {"formula": "ROA (as reported, percent)", "interpretation": "Profitability"},
    "roe_pct": {"formula": "ROE (as reported, percent)", "interpretation": "Profitability"},
    "net_interest_margin_pct": {"formula": "NIMY (as reported, percent)", "interpretation": "Earnings on interest-bearing assets"},
    "brokered_deposits_to_deposits": {"formula": "BRO / DEP", "interpretation": "Funding dependence; BRO is brokered deposits"},
    "cash_to_assets": {"formula": "CHBAL / ASSET", "interpretation": "Balance-sheet liquidity proxy"},
    "loans_to_deposits": {"formula": "LNLSGR / DEP", "interpretation": "Funding liquidity / loan intensity"},
}
FEATURE_COLUMNS = list(FEATURE_DEFINITIONS)


def previous_quarter(yyyymmdd: str) -> str:
    """Previous actual calendar quarter-end, not previous observed row."""
    observed = pd.Timestamp(datetime.strptime(yyyymmdd, "%Y%m%d"))
    current = observed.to_period("Q")
    if current.end_time.normalize() != observed:
        raise ValueError(f"Not a valid calendar quarter-end: {yyyymmdd}")
    return (current - 1).end_time.normalize().strftime("%Y%m%d")


def chronological_bucket(repdte: str) -> str:
    if FIRST <= repdte <= TRAIN_END:
        return "train"
    if GAP_START <= repdte <= GAP_END:
        return "gap"
    if TEST_START <= repdte <= TEST_END:
        return "test"
    return "outside"


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Numerical, finite quotient; a missing/nonpositive denominator -> NaN."""
    num = pd.to_numeric(numerator, errors="coerce").astype("float64")
    den = pd.to_numeric(denominator, errors="coerce").astype("float64")
    valid = den.gt(0) & np.isfinite(den) & np.isfinite(num)
    out = pd.Series(np.nan, dtype="float64", index=num.index)
    out.loc[valid] = num.loc[valid] / den.loc[valid]
    return out.replace([np.inf, -np.inf], np.nan)


def make_financial_features(prior_quarter_panel: pd.DataFrame) -> pd.DataFrame:
    """Create predictors using ONLY a past quarter's financial columns."""
    needed = set(KEYS + RAW_FIELDS)
    absent = needed.difference(prior_quarter_panel.columns)
    if absent:
        raise ValueError(f"Missing required Part 2 financial columns: {sorted(absent)}")
    if forbidden := FORBIDDEN.intersection(prior_quarter_panel.columns):
        raise ValueError(f"Future-outcome/status columns found in financial panel: {sorted(forbidden)}")

    d = prior_quarter_panel[KEYS + RAW_FIELDS].copy()
    for col in RAW_FIELDS:
        d[col] = pd.to_numeric(d[col], errors="coerce").astype("float64")
        d[col] = d[col].replace([np.inf, -np.inf], np.nan)

    out = d[KEYS].copy()
    out["equity_to_assets"] = safe_divide(d.EQ, d.ASSET)
    out["tier1_leverage_pct"] = d.RBC1AAJ
    out["noncurrent_loans_to_loans"] = safe_divide(d.NCLNLS, d.LNLSGR)
    # These FDIC fields are dollar amounts, not precomputed percentages.
    out["pastdue90_nonaccrual_assets_to_assets"] = safe_divide(d.P9ASSET + d.NAASSET, d.ASSET)
    out["net_chargeoffs_to_loans"] = safe_divide(d.NTLNLS, d.LNLSGR)
    out["loan_loss_allowance_to_loans"] = safe_divide(d.LNATRES, d.LNLSGR)
    out["real_estate_loans_to_loans"] = safe_divide(d.LNRE, d.LNLSGR)
    out["efficiency_proxy"] = safe_divide(d.NONIX, d.INTINC - d.EINTEXP + d.NONII)
    out["roa_pct"] = d.ROA
    out["roe_pct"] = d.ROE
    out["net_interest_margin_pct"] = d.NIMY
    out["brokered_deposits_to_deposits"] = safe_divide(d.BRO, d.DEP)
    out["cash_to_assets"] = safe_divide(d.CHBAL, d.ASSET)
    out["loans_to_deposits"] = safe_divide(d.LNLSGR, d.DEP)
    return out.replace([np.inf, -np.inf], np.nan)


def make_preprocessor() -> Pipeline:
    """Unfitted factory: instantiate inside EVERY Part 5 development CV fold."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])


def validate_label_frame(df: pd.DataFrame, quarter: str) -> pd.DataFrame:
    if not set(KEYS + [LABEL]).issubset(df.columns):
        raise ValueError(f"{quarter}: missing expected label keys/target")
    out = df[KEYS + [LABEL]].copy()
    out["REPDTE"] = out["REPDTE"].astype("string")
    if not out["REPDTE"].eq(quarter).all():
        raise ValueError(f"{quarter}: REPDTE does not match labelled file name")
    out["CERT"] = pd.to_numeric(out["CERT"], errors="raise").astype("int64")
    if out.duplicated(KEYS).any():
        raise ValueError(f"{quarter}: duplicate labelled CERT–REPDTE")
    out[LABEL] = pd.to_numeric(out[LABEL], errors="coerce")
    if not out[LABEL].dropna().isin([0, 1]).all():
        raise ValueError(f"{quarter}: labels must be 0, 1 or missing")
    return out


def validate_lagged_alignment(base: pd.DataFrame) -> None:
    if base.duplicated(KEYS).any():
        raise ValueError("Duplicate bank-quarter rows in model candidate table")
    if base[LABEL].isna().any() or not base[LABEL].isin([0, 1]).all():
        raise ValueError("Censored/invalid labels leaked into eligible model rows")
    expected = base.REPDTE.astype(str).map(previous_quarter)
    if not expected.eq(base[SOURCE_DATE].astype(str)).all():
        raise ValueError("LOOK-AHEAD GUARD: feature quarter is not exactly q-1")
    if not (base[SOURCE_DATE].astype(str) < base.REPDTE.astype(str)).all():
        raise ValueError("LOOK-AHEAD GUARD: features must strictly predate prediction quarter")
    if not all(pd.api.types.is_numeric_dtype(base[f]) for f in FEATURE_COLUMNS):
        raise ValueError("Non-numeric predictor created")
    if np.isinf(base[FEATURE_COLUMNS].to_numpy(dtype="float64")).any():
        raise ValueError("Infinite values remain in predictors")


def build_lagged_one_quarter(targets_at_q: pd.DataFrame, prior_financials: pd.DataFrame,
                             *, q: str, prior_q: str) -> tuple[pd.DataFrame, int]:
    """Pair target at q with *observed* financial rows from q-1 for SAME CERT.

    Do not forward/backward fill reporting gaps. Unmatched CERTs are excluded and
    separately counted; partial missing ratios remain NaN for train-only imputation.
    """
    if previous_quarter(q) != prior_q:
        raise ValueError("LOOK-AHEAD GUARD: supplied quarter is not the immediate past quarter")
    if not prior_financials.REPDTE.astype("string").eq(prior_q).all():
        raise ValueError("Financial source dates do not match the required lag")
    if prior_financials.duplicated(KEYS).any():
        raise ValueError("Duplicate CERT–REPDTE in past-quarter Financials")
    feats = make_financial_features(prior_financials).drop(columns="REPDTE")
    merged = targets_at_q.merge(feats, on="CERT", how="left", validate="one_to_one", indicator=True)
    no_history = int(merged._merge.eq("left_only").sum())
    merged = merged.loc[merged._merge.eq("both")].drop(columns="_merge").copy()
    merged[SOURCE_DATE] = prior_q
    merged[LABEL] = merged[LABEL].astype("int8")
    return merged[KEYS + [SOURCE_DATE] + FEATURE_COLUMNS + [LABEL]], no_history


def profile(df: pd.DataFrame) -> dict:
    y = df[LABEL]
    return {
        "rows": int(len(df)),
        "positive_bank_quarters": int(y.eq(1).sum()),
        "negative_bank_quarters": int(y.eq(0).sum()),
        "distinct_banks": int(df.CERT.nunique()),
        "distinct_positive_bank_CERTs": int(df.loc[y.eq(1), "CERT"].nunique()),
        "first_quarter": str(df.REPDTE.min()) if len(df) else None,
        "last_quarter": str(df.REPDTE.max()) if len(df) else None,
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run(root: Path, *, validate_only: bool = False) -> dict:
    from pyarrow.parquet import read_schema  # only actual I/O path needs pyarrow

    panel_dir = root / "data_processed" / "part2" / "panel"
    labels_dir = root / "data_processed" / "part2" / "labels"
    output_dir = root / "data_processed" / "part4"
    results_dir = root / "results" / "part4"
    if not panel_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("Part 2 panel/labels folders missing. Do NOT rerun Part 2 automatically.")
    label_files = sorted(labels_dir.glob("REPDTE=*.parquet"))
    panel_files = sorted(panel_dir.glob("REPDTE=*.parquet"))
    if len(label_files) != 170 or len(panel_files) != 170:
        raise ValueError(f"Expected 170 Part 2 quarters; found panel={len(panel_files)}, labels={len(label_files)}")
    if {p.name for p in label_files} != {p.name for p in panel_files}:
        raise ValueError("Part 2 panel/labels quarter file names differ")

    # Lightweight schema-only checks across Parquet footers; NO data scan.
    for p in panel_files:
        fields = set(read_schema(str(p)).names)
        if not set(KEYS + RAW_FIELDS).issubset(fields):
            raise ValueError(f"{p.name}: missing required columns: {sorted(set(KEYS + RAW_FIELDS) - fields)}")
        if prohibited := FORBIDDEN & fields:
            raise ValueError(f"{p.name}: detected forbidden future outcome columns: {sorted(prohibited)}")
    for p in label_files:
        fields = set(read_schema(str(p)).names)
        if not set(KEYS + [LABEL]).issubset(fields):
            raise ValueError(f"{p.name}: incomplete label schema")
    if validate_only:
        return {"status": "SCHEMA_OK", "quarter_files": 170, "no_outputs_written": True}

    by_label = {p.stem.split("=", 1)[1]: p for p in label_files}
    no_history = Counter()
    censored = Counter()
    label_counts_before_lag = Counter()
    group_frames: dict[str, list[pd.DataFrame]] = {"train": [], "test": []}
    for q in sorted(by_label):
        if q > TEST_END:
            continue
        bucket = chronological_bucket(q)
        if bucket == "outside":
            continue
        labels = validate_label_frame(pd.read_parquet(by_label[q], columns=KEYS + [LABEL]), q)
        censored[bucket] += int(labels[LABEL].isna().sum())
        labels = labels.loc[labels[LABEL].notna()].copy()
        label_counts_before_lag[bucket] += len(labels)
        if bucket == "gap":
            # The embargo period is never passed to a model or imputation fit.
            continue
        prior = previous_quarter(q)
        past_file = panel_dir / f"REPDTE={prior}.parquet"
        if not past_file.is_file():
            if q == FIRST:
                no_history[bucket] += len(labels)
                continue
            raise FileNotFoundError(f"{q}: missing q-1 Financials {past_file}; no interpolation permitted")
        past = pd.read_parquet(past_file, columns=KEYS + RAW_FIELDS)
        past["REPDTE"] = past["REPDTE"].astype("string")
        past["CERT"] = pd.to_numeric(past["CERT"], errors="raise").astype("int64")
        frame, without_history = build_lagged_one_quarter(labels, past, q=q, prior_q=prior)
        no_history[bucket] += without_history
        group_frames[bucket].append(frame)

    if not group_frames["train"] or not group_frames["test"]:
        raise ValueError("Empty eligible train/test split; check dates and Part 2 inputs")
    train = pd.concat(group_frames["train"], ignore_index=True)
    test = pd.concat(group_frames["test"], ignore_index=True)
    validate_lagged_alignment(train)
    validate_lagged_alignment(test)
    if not train.REPDTE.between(FIRST, TRAIN_END).all() or not test.REPDTE.between(TEST_START, TEST_END).all():
        raise AssertionError("Temporal split contamination")
    if not train.REPDTE.max() < test.REPDTE.min():
        raise AssertionError("Train and test overlap in chronology")
    distinct_test_failures = int(test.loc[test[LABEL].eq(1), "CERT"].nunique())
    if distinct_test_failures < MIN_UNIQUE_POSITIVE_TEST_BANKS:
        raise ValueError(
            f"Test has {distinct_test_failures} distinct positive-label CERTs, under required "
            f"{MIN_UNIQUE_POSITIVE_TEST_BANKS}; STOP and review split. Do NOT silently change dates."
        )

    # All feature selection and fitted preprocessing depend on TRAIN ONLY.
    selected = [f for f in FEATURE_COLUMNS if train[f].notna().any()]
    empty_in_train = [f for f in FEATURE_COLUMNS if f not in selected]
    if not selected:
        raise ValueError("All candidate predictors missing in the training period")
    preprocessing = make_preprocessor()
    preprocessing.fit(train[selected])  # NEVER fit on test or train+test.
    assert preprocessing.n_features_in_ == len(selected)
    if not np.isfinite(preprocessing.named_steps["imputer"].statistics_).all():
        raise ValueError("Nonfinite train-fitted medians")
    # Transformation-only sanity checks (no models, no test scoring or tuning).
    for split_name, df in (("train", train), ("test", test)):
        check = preprocessing.transform(df[selected].head(2000))
        if check.shape[1] != len(selected) or not np.isfinite(check).all():
            raise ValueError(f"{split_name}: preprocessing returned invalid output")

    split_audit = {
        "train_dates": [FIRST, TRAIN_END], "gap_dates": [GAP_START, GAP_END],
        "test_dates": [TEST_START, TEST_END], "one_quarter_financial_reporting_lag": True,
        "four_quarter_outcome_horizon": True,
        "test_minimum_distinct_positive_CERTs_required": MIN_UNIQUE_POSITIVE_TEST_BANKS,
        "test_meets_minimum": True,
        "training": profile(train), "test": profile(test),
        "labelled_rows_before_lag_by_bucket": dict(label_counts_before_lag),
        "censored_y_rows_excluded_by_bucket": dict(censored),
        "labelled_rows_without_observed_previous_quarter_by_bucket": dict(no_history),
        "gap_is_never_modelled": True,
        "note": "Distinct positive CERTs count banks with at least one positive held-out bank-quarter label; not positive rows."
    }
    missing = pd.DataFrame([
        {"feature": f, "nonmissing_train": int(train[f].notna().sum()),
         "missing_train": int(train[f].isna().sum()),
         "missing_train_pct": round(100 * train[f].isna().mean(), 4),
         "included_after_train_only_coverage_check": f in selected}
        for f in FEATURE_COLUMNS
    ])
    stats = preprocessing.named_steps["imputer"].statistics_
    scaler = preprocessing.named_steps["scaler"]
    pipeline_audit = {
        "pipeline_steps": ["SimpleImputer(strategy='median')", "RobustScaler()"],
        "fit_scope": "TRAIN ONLY (1984Q1 through 2007Q4); test used for transform-only smoke check",
        "selected_features_from_training_only": selected,
        "all_missing_training_features_dropped": empty_in_train,
        "imputer_train_medians": {f: float(v) for f, v in zip(selected, stats)},
        "robust_scaler_train_centers": {f: float(v) for f, v in zip(selected, scaler.center_)},
        "robust_scaler_train_scales": {f: float(v) for f, v in zip(selected, scaler.scale_)},
        "sklearn_version": sklearn.__version__, "pandas_version": pd.__version__,
        "test_labels_not_used_for_preprocessing": True,
        "part5_cv_warning": "Instantiate fresh preprocessing in each training fold (full sklearn Pipeline including estimator); never CV with this already-fitted artifact.",
    }
    feature_schema = {
        "feature_quarter_policy": "Always take source financial q-1 for target prediction quarter q; never backfill bank gaps",
        "release_vintage_limit": "Without actual filing/publication timestamps and historical vintages, even one-quarter lag is not a formal real-time availability guarantee.",
        "feature_definitions": FEATURE_DEFINITIONS,
        "feature_columns_selected_from_training": selected,
        "forbidden_columns": sorted(FORBIDDEN),
        "model_input_columns": selected,
        "never_model_input": KEYS + [SOURCE_DATE, LABEL],
        "schema_source": "FDIC BankFind financial field dictionary and 33-field acquisition allowlist",
    }

    # Write outputs only after all key audits pass; never touch Part 1–3 outputs.
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    for split, df in (("train", train), ("test", test)):
        df[KEYS + [SOURCE_DATE] + selected].to_parquet(output_dir / f"{split}_features.parquet", index=False)
        df[KEYS + [LABEL]].to_parquet(output_dir / f"{split}_targets.parquet", index=False)
    joblib.dump(preprocessing, output_dir / "preprocessor_train_only.joblib")
    write_json(results_dir / "split_audit.json", split_audit)
    write_json(results_dir / "pipeline_audit.json", pipeline_audit)
    write_json(results_dir / "feature_schema.json", feature_schema)
    missing.to_csv(results_dir / "train_feature_missingness.csv", index=False)
    return {
        "status": "PASS", "train_rows": len(train), "test_rows": len(test),
        "test_distinct_positive_CERTs": distinct_test_failures,
        "features": selected, "dropped_features": empty_in_train,
        "output_dir": str(output_dir), "results_dir": str(results_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="FDIC project root")
    parser.add_argument("--validate-only", action="store_true", help="Check 170 quarterly schemas only; write no outputs")
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), validate_only=args.validate_only), indent=2, default=str))


if __name__ == "__main__":
    main()
