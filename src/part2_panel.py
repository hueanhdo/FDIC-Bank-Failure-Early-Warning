"""Part 2: quality audit, bank-quarter panel, and four-quarter failure labels."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# These codes identify merger CANDIDATES, not verified merger events.
# Resolve their meaning against FDIC history before final classification.
MERGER_CANDIDATE_CODES = {221, 222, 223, 224, 225}
NUMERIC_EXCEPT = {"CERT", "REPDTE", "STNAME"}
RATIO_FIELDS = {"ROA", "ROE", "NIMY"}
LAST_FULL_LABEL_QUARTER = date(2025, 6, 30)


# 1. Normalize identifiers and reporting dates
def as_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().replace({
        "": pd.NA, "99999999": pd.NA, "9999-12-31": pd.NA, "9999/12/31": pd.NA,
        "12/31/9999": pd.NA, "12-31-9999": pd.NA
    })
    digits = text.str.fullmatch(r"\d{8}").fillna(False)
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    out.loc[digits] = pd.to_datetime(text.loc[digits], format="%Y%m%d", errors="coerce")
    out.loc[~digits] = pd.to_datetime(text.loc[~digits], format="mixed", errors="coerce")
    return out


def as_cert(values: pd.Series) -> pd.Series:
    ids = pd.to_numeric(values, errors="coerce")
    if ids.isna().any() or (ids % 1).ne(0).any():
        raise ValueError("Invalid/missing CERT identifiers; inspect raw FDIC records")
    return ids.astype("int64")


def quarter_list(start: date, end: date) -> list[str]:
    return [f"{year}{month:02d}{day:02d}"
            for year in range(start.year, end.year + 1)
            for month, day in ((3, 31), (6, 30), (9, 30), (12, 31))
            if start <= date(year, month, day) <= end]


def quarter_index(q: str) -> int:
    return int(q[:4]) * 4 + int(q[4:6]) // 3 - 1


# 2. Reconcile FDIC failure versus assistance events (audit only)
def make_events(institutions: pd.DataFrame, failures: pd.DataFrame,
                as_of: date = date(2026, 6, 30),
                study_start: date = date(1984, 3, 31)) -> pd.DataFrame:
    needed_inst = {"CERT", "ACTIVE", "CHANGEC1", "ENDEFYMD"}
    needed_fail = {"CERT", "FAILDATE", "RESTYPE"}
    if needed_inst - set(institutions.columns) or needed_fail - set(failures.columns):
        raise ValueError("Missing event columns in Institutions or Failures")

    inst = institutions[sorted(needed_inst)].copy()
    inst["CERT"] = as_cert(inst["CERT"])
    if inst["CERT"].duplicated().any():
        raise ValueError("Duplicate CERT in Institutions; review before any joins")

    # Audit all transactions; use only confirmed FAILURE events for the target.
    transactions = failures[["CERT", "FAILDATE", "RESTYPE"]].copy()
    transactions["failure_date"] = as_dates(transactions["FAILDATE"])
    transactions["RESTYPE"] = transactions["RESTYPE"].astype("string").str.strip().str.upper()
    if transactions["failure_date"].isna().any():
        raise ValueError("Invalid FAILDATE in FDIC event records")
    if (~transactions["RESTYPE"].isin(["FAILURE", "ASSISTANCE"])).any():
        raise ValueError("Unexpected RESTYPE: review before labelling")

    historical = transactions["failure_date"].lt(pd.Timestamp(study_start))
    missing_cert = pd.to_numeric(transactions["CERT"], errors="coerce").isna()
    if (missing_cert & ~historical).any():
        raise ValueError("Missing CERT within study-period FDIC events")

    in_scope = transactions.loc[~historical].copy()
    in_scope["CERT"] = as_cert(in_scope["CERT"])
    failures_only = in_scope.loc[in_scope["RESTYPE"].eq("FAILURE")].copy()
    multiple_dates = failures_only.groupby("CERT")["failure_date"].nunique()
    ambiguous = set(multiple_dates[multiple_dates.gt(1)].index.astype(int))
    confirmed = (failures_only.loc[~failures_only["CERT"].isin(ambiguous),
                                   ["CERT", "failure_date"]]
                 .drop_duplicates("CERT"))

    ambiguous_ids = pd.DataFrame({"CERT": sorted(ambiguous)})
    events = inst.merge(confirmed, on="CERT", how="outer", validate="one_to_one")
    events = events.merge(ambiguous_ids, on="CERT", how="outer", validate="one_to_one")
    events["end_date"] = as_dates(events["ENDEFYMD"])
    events["ambiguous_failure"] = events["CERT"].isin(ambiguous)
    change = pd.to_numeric(events["CHANGEC1"], errors="coerce")
    active_text = events["ACTIVE"].astype("string").str.strip().str.lower()
    inactive = (active_text.isin(["0", "false", "no", "n"]) |
                pd.to_numeric(active_text, errors="coerce").eq(0))
    failed_asof = events["failure_date"].notna() & events["failure_date"].le(pd.Timestamp(as_of))
    ended_asof = events["end_date"].notna() & events["end_date"].le(pd.Timestamp(as_of))

    events["eventual_outcome"] = "unresolved"
    events.loc[inactive & ended_asof & ~failed_asof, "eventual_outcome"] = "other_exit"
    events.loc[inactive & ended_asof & change.isin(MERGER_CANDIDATE_CODES) & ~failed_asof,
               "eventual_outcome"] = "merger_candidate"
    events.loc[failed_asof, "eventual_outcome"] = "failure"
    events.loc[events["ambiguous_failure"], "eventual_outcome"] = "ambiguous_failure"
    events["event_date"] = events["failure_date"].combine_first(events["end_date"])

    audit = {
        "all_transaction_rows": len(transactions),
        "historical_rows_excluded_from_event_lookup": int(historical.sum()),
        "historical_missing_cert_rows": int((historical & missing_cert).sum()),
        "study_period_transaction_rows": len(in_scope),
        "study_period_assistance_rows": int(in_scope["RESTYPE"].eq("ASSISTANCE").sum()),
        "study_period_failure_rows": len(failures_only),
        "ambiguous_failure_certs": sorted(ambiguous),
        "ambiguous_failure_cert_count": len(ambiguous),
        "confirmed_failure_certs": len(confirmed),
    }
    output = events[["CERT", "eventual_outcome", "event_date", "failure_date", "end_date",
                     "ambiguous_failure", "ACTIVE", "CHANGEC1"]].copy()
    output.attrs["failure_audit"] = audit
    return output


# 3. Audit raw quarter, then exclude only structurally unusable ASSET records
def clean_quarter(raw: pd.DataFrame, quarter: str):
    mandatory = {"CERT", "REPDTE", "ASSET", "DEP", "EQ"}
    if missing := mandatory - set(raw.columns):
        raise ValueError(f"{quarter}: missing Financials fields: {sorted(missing)}")
    df = raw.copy()
    df["CERT"] = as_cert(df["CERT"])
    reported = as_dates(df["REPDTE"])
    if reported.isna().any() or not reported.dt.strftime("%Y%m%d").eq(quarter).all():
        raise ValueError(f"{quarter}: invalid or mismatched REPDTE")
    df["REPDTE"] = quarter
    if df.duplicated(["CERT", "REPDTE"]).any():
        raise ValueError(f"{quarter}: duplicate CERT–REPDTE")

    non_numeric = {}
    for field in df.columns:
        if field in NUMERIC_EXCEPT:
            continue
        original = df[field].astype("string").str.strip().replace("", pd.NA)
        number = pd.to_numeric(original, errors="coerce").replace([np.inf, -np.inf], np.nan)
        non_numeric[field] = int((original.notna() & number.isna()).sum())
        df[field] = number.astype("float64")
    if "STNAME" in df:
        df["STNAME"] = df["STNAME"].astype("string").str.strip().replace("", pd.NA)

    field_rows = []
    for field in df.columns:
        values = df[field]
        record = {"year": int(quarter[:4]), "field": field, "rows": len(df),
                  "missing": int(values.isna().sum()), "non_numeric": non_numeric.get(field, 0),
                  "iqr_outliers": 0}
        if field in RATIO_FIELDS:
            nums = values.dropna()
            if len(nums) >= 10:
                low, high = nums.quantile([.25, .75])
                iqr = high - low
                if iqr > 0:
                    record["iqr_outliers"] = int(((nums < low - 1.5 * iqr) |
                                                   (nums > high + 1.5 * iqr)).sum())
        field_rows.append(record)

    bad_asset_missing = df["ASSET"].isna()
    bad_asset_nonpositive = df["ASSET"].le(0).fillna(False)
    exclude = bad_asset_missing | bad_asset_nonpositive
    flags = {"raw_rows": len(df), "excluded_asset_missing": int(bad_asset_missing.sum()),
             "excluded_asset_nonpositive": int(bad_asset_nonpositive.sum()),
             "retained_rows": int((~exclude).sum()),
             "negative_deposits": int(df["DEP"].lt(0).sum()),
             "negative_loans": int(df["LNLSGR"].lt(0).sum()) if "LNLSGR" in df else 0}
    return df.loc[~exclude].reset_index(drop=True), field_rows, flags


# 4. Label future failures; censor competing exits and unresolved failures
def make_labels(cert: pd.Series, quarter: str, events: pd.DataFrame,
                label_end: date = LAST_FULL_LABEL_QUARTER):
    lookup = events.set_index("CERT")
    failure_date = cert.map(lookup["failure_date"])
    end_date = cert.map(lookup["end_date"])
    ambiguous = cert.map(lookup["ambiguous_failure"]).fillna(False).astype(bool)
    q = pd.Timestamp(f"{quarter[:4]}-{quarter[4:6]}-{quarter[6:]}")
    horizon = q + pd.DateOffset(years=1)
    already_exited = (failure_date.notna() & failure_date.le(q)) | (
        end_date.notna() & end_date.le(q))
    future_failure = failure_date.gt(q) & failure_date.le(horizon)
    competing_exit = (end_date.gt(q) & end_date.le(horizon) & ~future_failure)
    eligible = ~(already_exited | ambiguous | competing_exit)
    label = pd.Series(pd.NA, index=cert.index, dtype="Int8")
    if q.date() <= label_end:
        label.loc[eligible] = future_failure.loc[eligible].astype("int8")
    counts = {"already_exited": int(already_exited.sum()),
              "ambiguous_failure": int((ambiguous & ~already_exited).sum()),
              "competing_exit": int((competing_exit & ~already_exited & ~ambiguous).sum())}
    return label, counts


# 5. Process chronological partitions and persist concise audit evidence
def run(raw_dir: Path, output_dir: Path, results_dir: Path,
        start: date = date(1984, 3, 31), end: date = date(2026, 6, 30),
        label_end: date = LAST_FULL_LABEL_QUARTER):
    quarters = quarter_list(start, end)
    source = raw_dir / "financials"
    missing = [q for q in quarters if not (source / f"REPDTE={q}.parquet").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} quarters, first: {missing[:5]}")
    events = make_events(pd.read_parquet(raw_dir / "institutions.parquet"),
                         pd.read_parquet(raw_dir / "failures.parquet"),
                         as_of=end, study_start=start)
    failure_audit = events.attrs["failure_audit"]

    panels = output_dir / "panel"
    labels = output_dir / "labels"
    for folder in (panels, labels, results_dir):
        folder.mkdir(parents=True, exist_ok=True)
    quality, excluded, gaps, labels_by_year = [], [], [], defaultdict(lambda: [0, 0])
    last_quarter = {}
    total_raw = total_kept = 0
    label_exclusions = defaultdict(int)
    banks_at_last_quarter = set()

    for q in quarters:
        raw = pd.read_parquet(source / f"REPDTE={q}.parquet")
        cleaned, q_quality, q_flags = clean_quarter(raw, q)
        quality.extend(q_quality)
        excluded.append({"quarter": q, **q_flags})
        total_raw += q_flags["raw_rows"]
        total_kept += len(cleaned)

        # Detect genuine reporting gaps from RAW presence, not asset-based exclusions.
        raw_ids = as_cert(raw["CERT"])
        if q == quarters[-1]:
            banks_at_last_quarter = set(raw_ids.astype(int).tolist())
        index = quarter_index(q)
        for bank in raw_ids.to_numpy():
            bank = int(bank)
            previous = last_quarter.get(bank)
            if previous and index - previous[0] > 1:
                gaps.append({"CERT": bank, "previous_quarter": previous[1],
                             "next_quarter": q, "missing_quarters": index - previous[0] - 1})
            last_quarter[bank] = (index, q)

        y, exclusions = make_labels(cleaned["CERT"], q, events, label_end)
        for key, value in exclusions.items():
            label_exclusions[key] += value
        year = int(q[:4])
        labels_by_year[year][0] += int(y.notna().sum())
        labels_by_year[year][1] += int(y.eq(1).sum())

        # Financial panel contains no future outcome fields; labels are separate.
        cleaned.to_parquet(panels / f"REPDTE={q}.parquet", index=False, compression="snappy")
        pd.DataFrame({"CERT": cleaned["CERT"], "REPDTE": cleaned["REPDTE"],
                      "y_fail_next4q": y}).to_parquet(
                          labels / f"REPDTE={q}.parquet", index=False, compression="snappy")
        print(f"{q}: {len(cleaned):,} retained / {q_flags['raw_rows']:,} raw; {int(y.eq(1).sum())} positives")

    quality_year = (pd.DataFrame(quality)
                    .groupby(["year", "field"], as_index=False)[["rows", "missing", "non_numeric", "iqr_outliers"]]
                    .sum())
    quality_year["missing_pct"] = (quality_year["missing"] * 100 / quality_year["rows"]).round(3)
    first_observed = (quality_year.loc[quality_year["rows"].gt(quality_year["missing"])]
                      .groupby("field")["year"].min())
    quality_year["first_observed_year"] = quality_year["field"].map(first_observed)
    quality_year["coverage_note"] = "observed_or_unresolved_missing"
    prior = quality_year["year"].lt(quality_year["first_observed_year"])
    quality_year.loc[prior, "coverage_note"] = "possible_structural_missing_verify_dictionary"
    quality_year.loc[quality_year["first_observed_year"].isna(), "coverage_note"] = "never_observed_verify_dictionary"
    quality_year["decade"] = quality_year["year"] // 10 * 10
    quality_year.to_csv(results_dir / "quality_by_year.csv", index=False)

    summary_decade = (quality_year.groupby(["decade", "field"], as_index=False)[["rows", "missing"]]
                      .sum())
    summary_decade["missing_pct"] = (100 * summary_decade["missing"] / summary_decade["rows"]).round(3)
    summary_decade.to_csv(results_dir / "missing_by_decade.csv", index=False)
    pd.DataFrame(excluded).to_csv(results_dir / "exclusions_and_invalid.csv", index=False)
    pd.DataFrame(gaps, columns=["CERT", "previous_quarter", "next_quarter", "missing_quarters"]).to_csv(
        results_dir / "reporting_gaps.csv", index=False)
    pd.DataFrame([{"year": year, "labelled_rows": v[0], "fail_next4q": v[1]}
                  for year, v in sorted(labels_by_year.items())]).to_csv(
                      results_dir / "labels_by_year.csv", index=False)
    # A bank is right-censored if it is still reporting in the final data quarter
    # and has no failure/exit event by that quarter (current ACTIVE alone is insufficient).
    cutoff = pd.Timestamp(end)
    no_exit_yet = events["event_date"].isna() | events["event_date"].gt(cutoff)
    present_at_end = events["CERT"].isin(banks_at_last_quarter)
    events.loc[present_at_end & no_exit_yet & ~events["ambiguous_failure"],
               "eventual_outcome"] = "right_censored"
    (events.groupby(["eventual_outcome", "CHANGEC1"], dropna=False).size()
     .rename("institutions").reset_index().to_csv(results_dir / "exit_code_audit.csv", index=False))
    events.to_parquet(output_dir / "event_lookup_AUDIT_ONLY.parquet", index=False)

    summary = {"quarters": len(quarters), "raw_rows": total_raw, "panel_rows": total_kept,
               "excluded_rows": total_raw - total_kept, "duplicate_cert_repdte": 0,
               "reporting_gaps": len(gaps), "label_exclusions": dict(label_exclusions),
               "failure_event_audit": failure_audit,
               "labelled_rows": sum(v[0] for v in labels_by_year.values()),
               "positive_labels": sum(v[1] for v in labels_by_year.values()),
               "events_by_outcome": events["eventual_outcome"].value_counts().to_dict(),
               "label_end": label_end.isoformat(),
               "note": "Merger codes/structural missingness are candidates pending FDIC dictionary review. Event data and labels are never predictors."}
    (results_dir / "part2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data_raw/fdic"))
    parser.add_argument("--out", type=Path, default=Path("data_processed/part2"))
    parser.add_argument("--results", type=Path, default=Path("results/part2"))
    args = parser.parse_args()
    print(json.dumps(run(args.raw, args.out, args.results), indent=2))


if __name__ == "__main__":
    main()
