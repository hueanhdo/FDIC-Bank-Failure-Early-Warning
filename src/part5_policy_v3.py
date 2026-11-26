"""Part 5 v3: development-only threshold repair and separately gated final evaluation.

Unit of evaluation: one bank-quarter observation (CERT, REPDTE). A predicted
alarm and its four-quarter future-failure target must belong to the SAME row.
Unlike the legacy bank-ever-max aggregation, this is invariant to the duration
of the evaluation period and does not credit a distant false alarm as timely.

Model choice and all fold fits remain frozen. Recalibration uses *only* the
already-saved development out-of-fold (OOF) predictions. It never reads test.
Final test is a separate, user-approved, one-shot command and is unavailable
until a valid v3 development selection has been written.

Cost convention: FDIC COST is a retrospective failure-resolution proxy, not
an observed cost of a missed early warning. False-alarm supervisory costs are
unobserved and set as explicit fractions of that proxy. The proxy cost is
charged PER BANK-QUARTER DECISION, so repeated pre-failure observations can
incur multiple losses. Totals are scenario indices, NOT actual FDIC expenses
or independent event-level losses. Compare test with OOF on the same unit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.part5_model import (
    KEYS, LABEL, TRAIN_END, TEST_START, TEST_END,
    _parse_costs, _read_split, make_model, require_inputs, score_metrics,
)

METHOD_VERSION = "V3_BANK_QUARTER_ALIGNED"
SCENARIO_FRACTIONS = (0.005, 0.01, 0.05)
COST_START = pd.Timestamp("1984-01-01")  # Whole 1984Q1, not just 31 March.
ORIGINAL_SELECTION = "development_selection.json"
V3_SELECTION = "development_selection_v3.json"
V3_COST_AUDIT = "cost_threshold_audit_v3.json"
V3_SWEEP = "cost_threshold_sweep_v3.csv"
V3_FINAL = "test_metrics_v3.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(2 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def dump_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def assert_test_unused(results: Path) -> None:
    # Never modify or delete these; a single attempted test disallows more work.
    for name in ("final_test_lock.json", "test_metrics.json", V3_FINAL,
                 "test_confusion_matrix_v3.csv"):
        if (results / name).exists():
            raise FileExistsError(f"STOP: held-out test already attempted or evaluated: {name}")


def validate_oof(oof: pd.DataFrame, chosen: str, fold_audit: pd.DataFrame) -> pd.DataFrame:
    fields = ["fold", "model", "CERT", "REPDTE", "y", "probability"]
    if not set(fields).issubset(oof.columns):
        raise ValueError(f"Missing development OOF fields: {sorted(set(fields)-set(oof.columns))}")
    d = oof[fields].copy()
    if d.empty or d.isna().any().any():
        raise ValueError("Development OOF contains empty/null rows")
    if not d["model"].eq(chosen).all():
        raise ValueError("OOF model does not match frozen development selection")
    if d.duplicated(["fold", "CERT", "REPDTE"]).any() or d.duplicated(KEYS).any():
        raise ValueError("Duplicate development OOF bank-quarter prediction")
    if not d["y"].isin([0, 1]).all():
        raise ValueError("OOF labels must be exactly 0 or 1")
    p = pd.to_numeric(d["probability"], errors="raise").to_numpy(dtype=float)
    if not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
        raise ValueError("OOF probabilities must be finite and in [0,1]")
    d["REPDTE"] = d["REPDTE"].astype(str)
    dates = pd.to_datetime(d["REPDTE"], format="%Y%m%d", errors="raise")
    if (dates.dt.to_period("Q").dt.end_time.dt.normalize() != dates).any():
        raise ValueError("OOF REPDTE must be calendar-quarter ends")
    if not d["REPDTE"].between("19840331", TRAIN_END).all():
        raise ValueError("LEAKAGE: OOF prediction outside development dates")
    if len(fold_audit) != 3 or set(d.fold.tolist()) != set(fold_audit.fold.tolist()):
        raise ValueError("Development OOF fold IDs differ from recorded three folds")
    for fold in fold_audit.itertuples(index=False):
        sub = d.loc[d.fold.eq(fold.fold)]
        if len(sub) != int(fold.val_rows) or int(sub.y.sum()) != int(fold.val_positive_rows):
            raise ValueError(f"OOF fold {fold.fold} differs from recorded validation sample")
        if not sub.REPDTE.between(str(fold.val_start), str(fold.val_end)).all():
            raise ValueError(f"OOF fold {fold.fold} dates extend beyond validation range")
    return d


def decision_counts(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    if not np.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("Threshold must be in [0,1]")
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if not len(y) or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Invalid bank-quarter outcome labels")
    if len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid bank-quarter probability vector")
    alarms = p >= t
    tp = int((alarms & (y == 1)).sum())
    fp = int((alarms & (y == 0)).sum())
    fn = int((~alarms & (y == 1)).sum())
    tn = int((~alarms & (y == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "bank_quarters": len(y)}


def optimal_threshold_exact(y: np.ndarray, p: np.ndarray, fp_fraction: float) -> dict:
    """Exact search over all distinct OOF scores, in O(n log n) time.

    Prediction rule p >= t. Select higher t on exact cost ties. The lowest
    possible cost is found without building an expensive n x n threshold grid.
    Cost is expressed in C_FN units so the monetary scale does not affect t.
    """
    if not (np.isfinite(fp_fraction) and 0 < fp_fraction < 1):
        raise ValueError("Invalid false-alarm cost fraction")
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if not len(y) or not set(np.unique(y)) == {0, 1}:
        raise ValueError("Both label classes must be present to select threshold")
    if len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid OOF predictions")
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    ends = np.flatnonzero(np.r_[ps[:-1] != ps[1:], True])
    fp = np.cumsum(ys == 0)[ends]
    tp = np.cumsum(ys == 1)[ends]
    fn = int(y.sum()) - tp
    thresholds = ps[ends]
    costs = fp_fraction * fp + fn
    # Include no-alert option when highest score is strictly below 1.
    if float(ps[0]) < 1.0:
        t_none = float(np.nextafter(ps[0], 1.0))
        thresholds = np.r_[thresholds, t_none]
        fp = np.r_[fp, 0]
        fn = np.r_[fn, int(y.sum())]
        costs = np.r_[costs, float(y.sum())]
    min_cost = float(costs.min())
    ties = np.flatnonzero(np.isclose(costs, min_cost, rtol=0, atol=1e-10))
    winner = ties[np.argmax(thresholds[ties])]
    t = float(thresholds[winner])
    counts = decision_counts(y, p, t)
    return {"fraction": float(fp_fraction), "threshold": t,
            **counts, "proxy_loss_in_c_fn_units": float(fp_fraction * counts["fp"] + counts["fn"])}


def cost_scenarios(y: np.ndarray, p: np.ndarray, c_fn: float,
                   fractions: tuple[float, ...]) -> tuple[list[dict], pd.DataFrame]:
    if not np.isfinite(c_fn) or c_fn <= 0:
        raise ValueError("Historical COST proxy must be positive")
    scenarios = []
    for frac in sorted(set(fractions)):
        result = optimal_threshold_exact(y, p, frac)
        result["c_fn_proxy_thousand_usd"] = float(c_fn)
        result["c_fp_assumed_thousand_usd"] = float(frac * c_fn)
        result["proxy_cost_thousand_usd"] = float(c_fn * result["proxy_loss_in_c_fn_units"])
        result["proxy_cost_per_1000_bank_quarters_thousand_usd"] = float(
            1000 * result["proxy_cost_thousand_usd"] / len(y))
        scenarios.append(result)
    # Bounded diagnostic curve only. Exact minimization used all unique scores.
    sample_grid = np.unique(np.r_[np.linspace(0.001, 0.999, 101),
                  np.quantile(p, np.linspace(0.001, 0.999, 101)),
                  [item["threshold"] for item in scenarios]])
    neg = np.sort(p[y == 0]); pos = np.sort(p[y == 1])
    fp_counts = len(neg) - np.searchsorted(neg, sample_grid, side="left")
    fn_counts = np.searchsorted(pos, sample_grid, side="left")
    curve = pd.concat([pd.DataFrame({
        "false_alarm_fraction": frac,
        "threshold": sample_grid,
        "fp_bank_quarters": fp_counts.astype(int),
        "fn_bank_quarters": fn_counts.astype(int),
        "proxy_cost_thousand_usd": c_fn * (frac * fp_counts + fn_counts),
    }) for frac in sorted(set(fractions))], ignore_index=True)
    return scenarios, curve


def estimate_1984_2007_cost(root: Path, fraction: float) -> dict:
    source = root / "data_raw" / "fdic" / "failures.parquet"
    if not source.is_file():
        raise FileNotFoundError(f"Required historic failure COST source missing: {source}")
    raw = pd.read_parquet(source, columns=["CERT", "FAILDATE", "RESTYPE", "COST"])
    # Reuse original parser for failure-only, ambiguity and positive COST rules.
    eligible = _parse_costs(raw)
    eligible = eligible.loc[eligible.FAILDATE.ge(COST_START)].copy()
    if len(eligible) < 10:
        raise ValueError("Insufficient 1984–2007 failure events with positive FDIC COST")
    median = float(eligible.cost_usd_thousands.median())
    return {
        "source": "data_raw/fdic/failures.parquet",
        "historical_cost_start": COST_START.strftime("%Y-%m-%d"),
        "historical_cost_end": "2007-12-31",
        "confirmed_failure_events_with_positive_cost": int(len(eligible)),
        "earliest_event": eligible.FAILDATE.min().strftime("%Y-%m-%d"),
        "latest_event": eligible.FAILDATE.max().strftime("%Y-%m-%d"),
        "c_fn_proxy_thousand_usd": median,
        "false_alarm_fraction_of_c_fn": float(fraction),
        "c_fp_assumed_thousand_usd": float(fraction * median),
        "false_alarm_cost_is_observed": False,
        "warning": "FDIC COST is a retrospective, potentially revised resolution-loss estimate, not observed avoided loss from a warning. False-alarm costs are unobserved. Amounts are not inflation-adjusted.",
    }


def recalibrate_development(root: Path) -> dict:
    root = root.resolve()
    out = root / "results" / "part5"
    private = root / "data_processed" / "part5"
    assert_test_unused(out)
    old = out / ORIGINAL_SELECTION
    oof_path = private / "chosen_model_cv_predictions.parquet"
    folds_path = out / "cv_fold_audit.csv"
    for path in (old, oof_path, folds_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen development evidence missing: {path}")
    targets = [out / V3_SELECTION, out / V3_COST_AUDIT, out / V3_SWEEP]
    if any(path.exists() for path in targets):
        raise FileExistsError("V3 development results already exist: no overwrite/recalibration loop")
    original = json.loads(old.read_text(encoding="utf-8"))
    if original.get("status") != "DEVELOPMENT_LOCKED_TEST_UNTOUCHED":
        raise ValueError("Original development selection not in frozen state")
    chosen = original.get("chosen_model")
    if chosen not in ("logistic", "hist_gradient_boosting"):
        raise ValueError("Unsupported original chosen model")
    frac = float(original["costs"]["false_alarm_fraction_of_c_fn"])
    if not np.isclose(frac, 0.01, rtol=0, atol=1e-12):
        raise ValueError("Unexpected primary cost scenario; do not change assumptions silently")
    folds = pd.read_csv(folds_path)
    oof = validate_oof(pd.read_parquet(oof_path), chosen, folds)
    y = oof.y.to_numpy(dtype=int); p = oof.probability.to_numpy(dtype=float)
    costs = estimate_1984_2007_cost(root, frac)
    scenarios, curve = cost_scenarios(y, p, costs["c_fn_proxy_thousand_usd"],
                                      SCENARIO_FRACTIONS)
    primary = next(row for row in scenarios if row["fraction"] == frac)
    cfg = dict(original)
    cfg.update({
        "chosen_threshold": primary["threshold"],
        "threshold_method_version": METHOD_VERSION,
        "threshold_selection": "Exact minimum of scenario-weighted FP/FN on each held-out development OOF BANK-QUARTER; no max-over-time grouping. Threshold learned using development only.",
        "threshold_proxy_cost_thousand_usd": primary["proxy_cost_thousand_usd"],
        "costs": costs,
        "source_original_selection_sha256": sha256_file(old),
        "source_oof_sha256": sha256_file(oof_path),
        "source_fold_audit_sha256": sha256_file(folds_path),
        "test_data_accessed": False,
        "final_test_approved": False,
        "model_fits_repeated": False,
        "status": "V3_DEVELOPMENT_SELECTION_FROZEN_TEST_UNTOUCHED",
    })
    audit = {
        "status": "V3_DEVELOPMENT_RECALIBRATION_COMPLETE_TEST_UNTOUCHED",
        "unit_of_policy": "(CERT, REPDTE) = one bank-quarter prediction",
        "target_horizon": "Failure within next four calendar quarters after prediction quarter q",
        "n_oof_bank_quarters": int(len(oof)),
        "n_oof_positive_bank_quarters": int(y.sum()),
        "n_unique_oof_banks": int(oof.CERT.nunique()),
        "n_fold_ids": int(oof.fold.nunique()),
        "costs": costs,
        "scenarios": scenarios,
        "chosen_model": chosen,
        "old_threshold": float(original["chosen_threshold"]),
        "v3_threshold": primary["threshold"],
        "not_event_level_loss": True,
        "test_data_accessed": False,
        "training_or_cv_rerun": False,
        "limitations": "Bank-quarter cost proxy may count multiple 4Q-window misses before one actual failure; figures do not represent total FDIC losses. Bank-specific intervention cost and historical publication vintages unavailable.",
    }
    # Write new V3-only files; selection last as completion marker. No mutation of V1/V2.
    out.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out / V3_SWEEP, index=False)
    dump_json(out / V3_COST_AUDIT, audit)
    dump_json(out / V3_SELECTION, cfg)
    return {"status": audit["status"], "chosen_model": chosen,
            "old_threshold": audit["old_threshold"], "new_threshold": primary["threshold"],
            "cv_retrained": False, "test_data_accessed": False,
            "historical_failure_events": costs["confirmed_failure_events_with_positive_cost"],
            "scenarios": scenarios}


def final_test_once(root: Path, *, approved: bool = False) -> dict:
    if not approved:
        raise PermissionError("STOP: --confirm-final-test requires explicit user approval")
    root = root.resolve()
    out = root / "results" / "part5"
    assert_test_unused(out)
    path = out / V3_SELECTION
    if not path.is_file():
        raise FileNotFoundError("V3 development selection missing; do not access held-out test")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("status") != "V3_DEVELOPMENT_SELECTION_FROZEN_TEST_UNTOUCHED":
        raise ValueError("Invalid v3 development selection status")
    if cfg.get("threshold_method_version") != METHOD_VERSION or cfg.get("test_data_accessed") is not False:
        raise ValueError("V3 time-aligned policy or test isolation not verified")
    original = out / ORIGINAL_SELECTION
    oof = root / "data_processed" / "part5" / "chosen_model_cv_predictions.parquet"
    folds = out / "cv_fold_audit.csv"
    if sha256_file(original) != cfg["source_original_selection_sha256"]:
        raise ValueError("Original development selection changed after V3 calibration")
    if sha256_file(oof) != cfg["source_oof_sha256"]:
        raise ValueError("Development OOF evidence changed after V3 calibration")
    if sha256_file(folds) != cfg["source_fold_audit_sha256"]:
        raise ValueError("Development fold audit changed after V3 calibration")
    cols, _ = require_inputs(root, include_test=False)  # no test observations read
    if cfg.get("training_feature_names") != cols:
        raise ValueError("Feature list differs from frozen development list")
    model_name = cfg.get("chosen_model")
    original_cfg = json.loads(original.read_text(encoding="utf-8"))
    if model_name != original_cfg.get("chosen_model"):
        raise ValueError("Model name changed after development")
    t = float(cfg["chosen_threshold"])
    if not (np.isfinite(t) and 0 <= t <= 1):
        raise ValueError("Invalid frozen threshold")
    c_fn = float(cfg["costs"]["c_fn_proxy_thousand_usd"])
    fraction = float(cfg["costs"]["false_alarm_fraction_of_c_fn"])
    if not (c_fn > 0 and np.isclose(fraction, 0.01)):
        raise ValueError("Unapproved cost assumptions")
    if (out / V3_FINAL).exists() or (out / "test_confusion_matrix_v3.csv").exists():
        raise FileExistsError("V3 final output already exists")
    # Final model fitting is development-only; do it BEFORE engaging test lock.
    # A training failure cannot waste the one-shot held-out test authorization.
    private = root / "data_processed" / "part5"
    if (private / "heldout_predictions_v3_LOCAL_ONLY.parquet").exists():
        raise FileExistsError("Local V3 held-out predictions already exist")
    train = _read_split(root, "train", cols)
    pipe = make_model(model_name)
    pipe.fit(train[cols], train[LABEL].to_numpy(dtype=int))
    # Create exclusive test lock immediately BEFORE reading test data.
    lock = out / "final_test_lock.json"
    out.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as f:
        dump = {"state": "V3_HELDOUT_TEST_ATTEMPTED_LOCKED",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "model": model_name, "threshold": t,
                "v3_selection_sha256": sha256_file(path),
                "instructions": "Do not remove, overwrite, or retry without new explicit user authorization."}
        json.dump(dump, f, indent=2)
    # Only below this line can the test be read. On failure, lock remains intact.
    test = _read_split(root, "test", cols)
    p = pipe.predict_proba(test[cols])[:, 1]
    y = test[LABEL].to_numpy(dtype=int)
    metrics = score_metrics(y, p, t)
    counts = decision_counts(y, p, t)
    assert (counts["tp"], counts["fp"], counts["fn"], counts["tn"]) == (
        metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"])
    proxy_cost = float(c_fn * (fraction * counts["fp"] + counts["fn"]))
    result = {
        "status": "V3_FINAL_TEST_COMPLETE_LOCKED_NO_REPEATS",
        "model": model_name,
        "threshold_frozen_from_development": t,
        "threshold_method_version": METHOD_VERSION,
        "bank_quarter_metrics": metrics,
        "test_distinct_positive_CERTs": int(test.loc[test[LABEL].eq(1), "CERT"].nunique()),
        "bank_quarter_proxy_cost_thousand_usd": proxy_cost,
        "proxy_cost_per_1000_bank_quarters_thousand_usd": float(1000 * proxy_cost / len(y)),
        "costs_frozen_from_development": cfg["costs"],
        "test_read_once_after_lock": True,
        "warning": "Cost proxy applies per prediction quarter, not independent failure event; no claims of observed FDIC supervisory savings or equivalent 23Q/66Q bank-level exposure.",
    }
    # Output only AFTER all evaluation steps; individual predictions stay local.
    private = root / "data_processed" / "part5"
    private.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"CERT": test.CERT.to_numpy(), "REPDTE": test.REPDTE.to_numpy(),
                  "y": y, "probability": p, "alarm": (p >= t).astype(int)}).to_parquet(
                      private / "heldout_predictions_v3_LOCAL_ONLY.parquet", index=False)
    pd.DataFrame([{"tn": metrics["tn"], "fp": metrics["fp"],
                   "fn": metrics["fn"], "tp": metrics["tp"]}]).to_csv(
                       out / "test_confusion_matrix_v3.csv", index=False)
    dump_json(out / V3_FINAL, result)
    return {"status": result["status"], "model": model_name,
            "test_AP": metrics["average_precision"], "precision": metrics["precision"],
            "recall": metrics["recall"], "f1": metrics["f1"],
            "n_test_bank_quarters": metrics["observations"], "no_repeats": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["recalibrate", "final-test"], required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--confirm-final-test", action="store_true")
    a = parser.parse_args()
    if a.phase == "recalibrate":
        if a.confirm_final_test:
            raise ValueError("Confirmation flag not valid in recalibration phase")
        result = recalibrate_development(a.root)
    else:
        result = final_test_once(a.root, approved=a.confirm_final_test)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
