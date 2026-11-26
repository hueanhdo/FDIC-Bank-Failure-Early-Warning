"""FDIC Part 5: purged quarterly development CV, bank-level cost policy, one-shot test.

Only load Part 4's model_input_columns as X. No test file is read in the
'validate' or 'development' phases. Final test requires a separate explicit
approval and is guarded by a pre-evaluation lock file.

FDIC COST is a retrospective estimated *resolution* loss, not a directly
observed supervisory missed-alarm cost. False-alarm review cost is NOT in FDIC
and is therefore explicitly assumed as a fraction of the development-only
median positive historical COST; sensitivity scenarios are reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.part4_features import make_preprocessor, LABEL, FEATURE_COLUMNS

KEYS = ["CERT", "REPDTE"]
SOURCE_DATE = "SOURCE_REPDTE"
TRAIN_END = "20071231"
TEST_START, TEST_END = "20090331", "20250630"
CV_SPLITS = 3
GAP_QUARTERS = 4
SEED = 42
# Explicit scenario, not an observed FDIC false-alarm cost.
DEFAULT_FP_FRACTION = 0.01
FP_SENSITIVITY = (0.005, 0.01, 0.05)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def require_inputs(root: Path, *, include_test: bool = False) -> tuple[list[str], Path]:
    """Validate file existence/schema metadata without reading the test observations."""
    stage4 = root / "data_processed" / "part4"
    result4 = root / "results" / "part4"
    features_manifest = result4 / "feature_schema.json"
    split_manifest = result4 / "split_audit.json"
    if not features_manifest.is_file() or not split_manifest.is_file():
        raise FileNotFoundError("Part 4 audit manifests missing; do not regenerate automatically")
    feat_spec = json.loads(features_manifest.read_text(encoding="utf-8"))
    split_spec = json.loads(split_manifest.read_text(encoding="utf-8"))
    cols = feat_spec.get("model_input_columns")
    if not isinstance(cols, list) or not cols or len(cols) != len(set(cols)):
        raise ValueError("Invalid model_input_columns in Part 4 feature schema")
    if not set(cols).issubset(FEATURE_COLUMNS):
        raise ValueError("Model input columns include non-approved future/identifier fields")
    if split_spec.get("train_dates") != ["19840331", TRAIN_END]:
        raise ValueError("Unexpected Part 4 development period")
    if split_spec.get("gap_dates") != ["20080331", "20081231"]:
        raise ValueError("Unexpected Part 4 purge gap")
    if split_spec.get("test_dates") != [TEST_START, TEST_END]:
        raise ValueError("Unexpected Part 4 held-out period")
    fnames = ["train_features.parquet", "train_targets.parquet"]
    if include_test:
        fnames += ["test_features.parquet", "test_targets.parquet"]
    for name in fnames:
        if not (stage4 / name).is_file():
            raise FileNotFoundError(f"Missing Part 4 output: {stage4/name}")
    return cols, stage4


def _read_split(root: Path, split: str, cols: list[str]) -> pd.DataFrame:
    """Only the approved feature columns and audit keys enter the returned frame."""
    root_data = root / "data_processed" / "part4"
    inputs = pd.read_parquet(root_data / f"{split}_features.parquet",
                             columns=KEYS + [SOURCE_DATE] + cols)
    targets = pd.read_parquet(root_data / f"{split}_targets.parquet",
                              columns=KEYS + [LABEL])
    for name, df in (("features", inputs), ("targets", targets)):
        if df.duplicated(KEYS).any():
            raise ValueError(f"Duplicate {split} {name} CERT-REPDTE keys")
    merged = inputs.merge(targets, on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(inputs) or len(merged) != len(targets):
        raise ValueError(f"{split}: unmatched feature-target rows")
    for d in ("CERT",):
        merged[d] = pd.to_numeric(merged[d], errors="raise").astype("int64")
    merged.REPDTE = merged.REPDTE.astype("string")
    merged[SOURCE_DATE] = merged[SOURCE_DATE].astype("string")
    if merged[LABEL].isna().any() or not merged[LABEL].isin([0, 1]).all():
        raise ValueError(f"{split}: missing or invalid labels")
    if split == "train" and not merged.REPDTE.le(TRAIN_END).all():
        raise ValueError("Development includes post-2007 observation")
    if split == "test" and not merged.REPDTE.between(TEST_START, TEST_END).all():
        raise ValueError("Held-out test includes out-of-period observation")
    # Quarter lag checks do not look at labels beyond the observation itself.
    prediction = pd.to_datetime(merged.REPDTE, format="%Y%m%d", errors="raise")
    source = pd.to_datetime(merged[SOURCE_DATE], format="%Y%m%d", errors="raise")
    if not (source.dt.to_period("Q") + 1).eq(prediction.dt.to_period("Q")).all():
        raise ValueError("LOOK-AHEAD: source financials are not precisely q-1")
    if not source.lt(prediction).all():
        raise ValueError("LOOK-AHEAD: financial source date is not before prediction quarter")
    if not all(pd.api.types.is_numeric_dtype(merged[f]) for f in cols):
        raise ValueError("Non-numeric model feature")
    if np.isinf(merged[cols].to_numpy(dtype=float)).any():
        raise ValueError("Infinite model inputs")
    return merged.sort_values(["REPDTE", "CERT"], kind="mergesort").reset_index(drop=True)


def quarter_folds(dates: pd.Series | np.ndarray, n_splits: int = CV_SPLITS,
                  gap: int = GAP_QUARTERS):
    """TimeSeriesSplit applied to *unique quarters*, never raw bank rows."""
    quarter_list = np.sort(pd.Series(dates).astype(str).unique())
    if len(quarter_list) < (n_splits + 1) * (gap + 2):
        raise ValueError("Too few distinct quarters for safe purged validation")
    for fold, (a, b) in enumerate(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(quarter_list), 1):
        train_q, val_q = quarter_list[a], quarter_list[b]
        if np.intersect1d(train_q, val_q).size:
            raise ValueError("Overlapping train/validation quarters")
        # Exactly 'gap' excluded whole quarters between folds.
        end_train_ix, start_val_ix = int(a[-1]), int(b[0])
        if start_val_ix - end_train_ix - 1 < gap:
            raise ValueError("Insufficient purge gap between train and validation")
        end = pd.Period(pd.to_datetime(train_q[-1], format="%Y%m%d"), freq="Q")
        first_val = pd.Period(pd.to_datetime(val_q[0], format="%Y%m%d"), freq="Q")
        if end + 4 >= first_val:
            raise ValueError("LOOK-AHEAD: a training label horizon reaches validation")
        yield fold, train_q, val_q


def make_classifier(name: str):
    if name == "logistic":
        return LogisticRegression(class_weight="balanced", max_iter=400,
                                  solver="lbfgs", random_state=SEED)
    if name == "hist_gradient_boosting":
        # Do not allow sklearn to invent a random within-fit validation split.
        return HistGradientBoostingClassifier(max_iter=55, max_leaf_nodes=15,
                min_samples_leaf=100, learning_rate=0.08, early_stopping=False,
                l2_regularization=0.1, random_state=SEED)
    raise ValueError(f"Unsupported model: {name}")


def make_model(name: str) -> Pipeline:
    # CRITICAL: instantiate this anew each time; never load Part 4 joblib.
    return Pipeline([("preprocessor", make_preprocessor()),
                     ("classifier", make_classifier(name))])


def score_metrics(y_true, proba: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    if y.size == 0 or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Empty or invalid labels")
    if not (np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()):
        raise ValueError("Invalid predicted probabilities")
    if np.unique(y).size != 2:
        raise ValueError("Both labels must be present for robust AP/ROC evaluation")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    return {
        "observations": int(len(y)), "positives": int(y.sum()),
        "positive_prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }


def fold_cv(data: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q = data.REPDTE.astype(str)
    metrics, fold_audit, oofs = [], [], []
    for fold, tr_quarters, va_quarters in quarter_folds(q):
        is_tr = q.isin(tr_quarters).to_numpy()
        is_va = q.isin(va_quarters).to_numpy()
        train, valid = data.loc[is_tr], data.loc[is_va]
        if not len(train) or not len(valid):
            raise ValueError("Empty CV fold")
        if train[LABEL].nunique() != 2 or valid[LABEL].nunique() != 2:
            raise ValueError("A CV fold has only one label; stop and review split")
        row = {
            "fold": fold,
            "train_start": str(tr_quarters[0]), "train_end": str(tr_quarters[-1]),
            "val_start": str(va_quarters[0]), "val_end": str(va_quarters[-1]),
            "train_quarters": len(tr_quarters), "val_quarters": len(va_quarters),
            "purge_quarters": GAP_QUARTERS,
            "train_rows": len(train), "val_rows": len(valid),
            "train_positive_rows": int(train[LABEL].sum()),
            "val_positive_rows": int(valid[LABEL].sum()),
            "val_positive_banks": int(valid.loc[valid[LABEL] == 1, "CERT"].nunique()),
        }
        fold_audit.append(row)
        for name in ("logistic", "hist_gradient_boosting"):
            model = make_model(name)
            model.fit(train[feature_cols], train[LABEL].to_numpy(dtype=int))
            p = model.predict_proba(valid[feature_cols])[:, 1]
            record = score_metrics(valid[LABEL], p, 0.5)
            metrics.append({"fold": fold, "model": name,
                            "val_start": row["val_start"], "val_end": row["val_end"], **record})
            oofs.append(pd.DataFrame({
                "fold": fold, "model": name, "CERT": valid.CERT.to_numpy(),
                "REPDTE": valid.REPDTE.to_numpy(),
                "y": valid[LABEL].to_numpy(dtype=int), "probability": p,
            }))
    return pd.DataFrame(metrics), pd.DataFrame(fold_audit), pd.concat(oofs, ignore_index=True)


def choose_model(cv: pd.DataFrame) -> tuple[str, dict]:
    means = cv.groupby("model", sort=True)["average_precision"].mean().to_dict()
    if set(means) != {"logistic", "hist_gradient_boosting"}:
        raise ValueError("Missing model CV results")
    # Fixed deterministic tie-break: simpler Logistic Regression.
    name = sorted(means, key=lambda k: (-means[k], 0 if k == "logistic" else 1))[0]
    return name, {k: float(v) for k, v in means.items()}


def _parse_costs(failure: pd.DataFrame) -> pd.DataFrame:
    expected = {"CERT", "FAILDATE", "RESTYPE", "COST"}
    if not expected.issubset(failure.columns):
        raise ValueError(f"FDIC failures missing {sorted(expected-set(failure.columns))}")
    x = failure[["CERT", "FAILDATE", "RESTYPE", "COST"]].copy()
    x.RESTYPE = x.RESTYPE.astype("string").str.upper().str.strip()
    x = x.loc[x.RESTYPE.eq("FAILURE")].copy()  # assistance is not failure
    x.FAILDATE = pd.to_datetime(x.FAILDATE, errors="coerce", format="mixed")
    x = x.loc[x.FAILDATE.notna() & (x.FAILDATE <= pd.Timestamp(TRAIN_END))]
    x.CERT = pd.to_numeric(x.CERT, errors="coerce")
    x = x.loc[x.CERT.notna()].copy()
    # Historical CERT with contradictory failure dates cannot calibrate costs.
    ambiguous = x.groupby("CERT").FAILDATE.nunique()
    x = x.loc[~x.CERT.isin(ambiguous[ambiguous > 1].index)].copy()
    raw_cost = x.COST.astype("string").str.replace(r"[,\$\s]", "", regex=True)
    x["cost_usd_thousands"] = pd.to_numeric(raw_cost, errors="coerce")
    x = x.loc[x.cost_usd_thousands.gt(0) & np.isfinite(x.cost_usd_thousands)]
    x = x.drop_duplicates(subset=["CERT"], keep="first")
    return x


def estimate_development_cost(root: Path, false_alarm_fraction: float) -> dict:
    if not (0 < false_alarm_fraction < 1):
        raise ValueError("False-alarm fraction must be strictly between zero and one")
    path = root / "data_raw" / "fdic" / "failures.parquet"
    if not path.is_file():
        raise FileNotFoundError("Historical development FDIC COST file missing: data_raw/fdic/failures.parquet")
    failures = pd.read_parquet(path, columns=["CERT", "FAILDATE", "RESTYPE", "COST"])
    x = _parse_costs(failures)
    if len(x) < 10:
        raise ValueError("Fewer than 10 valid historical failure COST observations; stop for review")
    median_cost = float(x.cost_usd_thousands.median())
    return {
        "fdic_cost_field": "COST",
        "fdic_loss_unit": "thousands of USD, as reported (retrospective estimates)",
        "historical_failures_used": len(x),
        "earliest_failure_date": x.FAILDATE.min().strftime("%Y-%m-%d"),
        "latest_failure_date": x.FAILDATE.max().strftime("%Y-%m-%d"),
        "historical_failure_cutoff": TRAIN_END,
        "c_fn_proxy_thousand_usd": median_cost,
        "false_alarm_fraction_of_c_fn": float(false_alarm_fraction),
        "c_fp_assumed_thousand_usd": float(false_alarm_fraction * median_cost),
        "false_alarm_cost_is_observed": False,
        "warning": "COST is a retrospectively reported FDIC resolution-loss estimate, NOT observed missed-warning harm; false-alarm inspection cost is unavailable and is an explicit scenario assumption. No test events used.",
    }


def summarize_bank_policy(scores: pd.DataFrame) -> pd.DataFrame:
    """At most one alert/cost per bank per validation fold.

    Policy summary, not a point-in-time row-level forecast: each (fold, CERT)
    aggregates whether a positive label/alert was ever observed within the fold.
    """
    needed = {"fold", "CERT", "y", "probability"}
    if not needed.issubset(scores):
        raise ValueError("Policy scores missing required fields")
    policy = scores.groupby(["fold", "CERT"], as_index=False, sort=False).agg(
        y=("y", "max"), probability=("probability", "max"))
    return policy


def cost_sweep(policy: pd.DataFrame, c_fn: float, fractions=FP_SENSITIVITY,
               preferred_fraction=DEFAULT_FP_FRACTION) -> tuple[float, pd.DataFrame]:
    if c_fn <= 0 or policy.y.nunique() != 2:
        raise ValueError("Invalid development cost baseline or single-class validation")
    scores = policy.probability.to_numpy(dtype=float)
    labels = policy.y.to_numpy(dtype=int)
    grid = np.unique(np.r_[np.linspace(0.001, 0.999, 201),
                           np.quantile(scores, np.linspace(0.001, 0.999, 101))])
    fractions = sorted(set((*fractions, float(preferred_fraction))))
    rows = []
    for fraction in fractions:
        c_fp = fraction * c_fn
        for t in grid:
            flags = scores >= t
            fp = int(np.sum(flags & (labels == 0)))
            fn = int(np.sum(~flags & (labels == 1)))
            rows.append({"false_alarm_fraction": fraction, "threshold": float(t),
                         "fp_bank_folds": fp, "fn_bank_folds": fn,
                         "total_proxy_cost_thousand_usd": float(c_fp*fp + c_fn*fn)})
    table = pd.DataFrame(rows)
    primary = table.loc[np.isclose(table.false_alarm_fraction, preferred_fraction)]
    # deterministic tie-break: prefer higher threshold to reduce false alarms
    chosen = primary.sort_values(["total_proxy_cost_thousand_usd", "threshold"],
                                 ascending=[True, False], kind="mergesort").iloc[0]
    return float(chosen.threshold), table


def development(root: Path, *, false_alarm_fraction: float = DEFAULT_FP_FRACTION) -> dict:
    out = root / "results" / "part5"
    if (out / "development_selection.json").exists() or (out / "final_test_lock.json").exists():
        raise FileExistsError("Part 5 development already locked. Never rerun automatically.")
    cols, _ = require_inputs(root, include_test=False)
    # Full development requires historical COST; validate its availability BEFORE CV.
    costs = estimate_development_cost(root, false_alarm_fraction)
    data = _read_split(root, "train", cols)
    if data.REPDTE.min() < "19840331" or data.REPDTE.max() != TRAIN_END:
        raise ValueError("Unexpected development dates")
    metrics, splits, oof = fold_cv(data, cols)
    best, mean_ap = choose_model(metrics)
    selected_oof = oof.loc[oof.model.eq(best)].copy()
    policy = summarize_bank_policy(selected_oof)
    threshold, sweep = cost_sweep(policy, costs["c_fn_proxy_thousand_usd"],
                                  preferred_fraction=false_alarm_fraction)
    primary = sweep.loc[np.isclose(sweep.false_alarm_fraction, false_alarm_fraction)]
    best_cost = primary.loc[np.isclose(primary.threshold, threshold)].iloc[0]
    cfg = {
        "status": "DEVELOPMENT_LOCKED_TEST_UNTOUCHED",
        "chosen_model": best,
        "model_average_precision_per_fold_mean": mean_ap,
        "chosen_threshold": threshold,
        "selection_metric": "mean fold-level average_precision_score (AP)",
        "threshold_selection": "minimize development out-of-fold proxy loss per (validation fold, CERT) with max score, maximum one alarm per bank-fold; not an ex-ante bank-quarter decision metric",
        "threshold_proxy_cost_thousand_usd": float(best_cost.total_proxy_cost_thousand_usd),
        "costs": costs,
        "n_folds": CV_SPLITS, "purge_quarters": GAP_QUARTERS,
        "training_feature_names": cols,
        "random_state": SEED,
        "test_file_access_during_development": False,
        "test_used_for_model_or_threshold_selection": False,
        "software_versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__},
        "notes": "No hyperparameter search. The one-time final test requires distinct authorized command and creates a lock before reading test data.",
    }
    out.mkdir(parents=True, exist_ok=True)
    private = root / "data_processed" / "part5"
    private.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "cv_metrics.csv", index=False)
    splits.to_csv(out / "cv_fold_audit.csv", index=False)
    selected_oof.to_parquet(private / "chosen_model_cv_predictions.parquet", index=False)
    sweep.to_csv(out / "cost_threshold_sweep.csv", index=False)
    save_json(out / "cost_threshold_audit.json", {
        "threshold": threshold, "scenarios": list(sorted(sweep.false_alarm_fraction.unique())),
        "costs": costs, "bank_fold_decisions": len(policy),
        "selected_cost_thousand_usd": float(best_cost.total_proxy_cost_thousand_usd),
        "limitations": cfg["threshold_selection"],
    })
    # Selection JSON is written LAST; this is the readiness/freeze marker.
    save_json(out / "development_selection.json", cfg)
    return {"status": cfg["status"], "model": best, "threshold": threshold,
            "mean_ap": mean_ap, "cv_folds": CV_SPLITS,
            "development_rows": len(data), "test_data_read": False}


def final_test(root: Path, *, confirm: bool = False) -> dict:
    """Disabled legacy evaluator: its ever-max bank aggregation is invalid.

    Do not use the old CLI final-test command. After reviewing development v3,
    use src.part5_policy_v3 with explicit approval for one held-out evaluation.
    """
    raise PermissionError(
        "LEGACY FINAL TEST DISABLED: bank-level max(y)/max(p) mismatches quarters. "
        "Use separately reviewed src.part5_policy_v3 final-test after approval."
    )


def validate_only(root: Path) -> dict:
    cols, directory = require_inputs(root, include_test=False)
    from pyarrow.parquet import read_schema
    for name in ("train_features.parquet", "train_targets.parquet"):
        file = directory / name
        fields = set(read_schema(str(file)).names)
        required = set(KEYS + ([SOURCE_DATE] + cols if "features" in name else [LABEL]))
        if not required.issubset(fields):
            raise ValueError(f"Missing Part 4 fields in {name}: {sorted(required-fields)}")
    fpath = root / "data_raw" / "fdic" / "failures.parquet"
    if not fpath.is_file():
        raise FileNotFoundError("Required historical FDIC failures for dev-only COST proxy missing")
    fields = set(read_schema(str(fpath)).names)
    if not {"CERT", "FAILDATE", "RESTYPE", "COST"}.issubset(fields):
        raise ValueError("FDIC failure COST metadata missing")
    return {"status": "SCHEMA_OK_NO_TEST_READ", "feature_columns": len(cols),
            "development_files_checked": 2, "test_files_accessed": False,
            "historical_failure_cost_source": str(fpath)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["validate", "development", "final-test"], required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--false-alarm-fraction", type=float, default=DEFAULT_FP_FRACTION)
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == "validate":
        result = validate_only(root)
    elif args.phase == "development":
        result = development(root, false_alarm_fraction=args.false_alarm_fraction)
    else:
        result = final_test(root, confirm=args.confirm_final_test)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
