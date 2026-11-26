"""Focused synthetic checks for v3 time alignment, leak guards and one-shot lock."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.part5_policy_v3 import (
    METHOD_VERSION, cost_scenarios, decision_counts, estimate_1984_2007_cost,
    final_test_once, optimal_threshold_exact, recalibrate_development,
    validate_oof,
)
from src.part5_model import final_test as disabled_legacy_final_test


def test_quarter_aligned_cost_cannot_pair_distant_false_alarm_with_failure():
    # Prior failure in distant future is not credited by an earlier y=0 alarm.
    y = np.array([0, 1, 0, 0])
    p = np.array([0.85, 0.005, 0.05, 0.05])
    at = decision_counts(y, p, 0.1)
    assert (at["tp"], at["fp"], at["fn"], at["tn"]) == (0, 1, 1, 2)


def test_cost_counts_consistent_with_observation_level_confusion():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.8, 0.1, 0.9, 0.3, 0.01])
    at = decision_counts(y, p, 0.5)
    assert at == {"tp": 1, "fp": 1, "fn": 1, "tn": 2, "bank_quarters": 5}


def test_exact_threshold_matches_bruteforce_all_unique_scores():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([.01, .02, .03, .02, .04, .99, .2, .6, .1])
    f = .03
    answer = optimal_threshold_exact(y, p, f)
    grid = np.unique(np.r_[p, float(np.nextafter(p.max(), 1.0))])
    costs = np.array([f * decision_counts(y, p, t)["fp"] + decision_counts(y, p, t)["fn"]
                      for t in grid])
    assert answer["proxy_loss_in_c_fn_units"] == pytest.approx(float(costs.min()))
    assert answer["threshold"] == pytest.approx(float(max(grid[np.isclose(costs, costs.min())])))


def test_three_cost_scenarios_without_model_rerun():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.01, 0.23, 0.37, 0.74, 0.9, 0.12])
    scenarios, curve = cost_scenarios(y, p, 2000., (.005, .01, .05))
    assert {x["fraction"] for x in scenarios} == {.005, .01, .05}
    assert set(curve.false_alarm_fraction.unique()) == {.005, .01, .05}
    for x in scenarios:
        assert x["proxy_cost_thousand_usd"] == pytest.approx(
            2000 * (x["fraction"] * x["fp"] + x["fn"]))


def test_oof_rejects_date_after_development_and_duplicate_bank_quarter():
    a = pd.DataFrame({"fold": [1, 1], "model": ["logistic"]*2,
       "CERT": [42, 43], "REPDTE": ["20070331", "20090331"],
       "y": [0, 1], "probability": [.2,.6]})
    folds = pd.DataFrame({"fold":[1,2,3], "val_rows":[2,1,1],
        "val_positive_rows":[1,0,0], "val_start":["20070331"]*3,
        "val_end":["20071231"]*3})
    with pytest.raises(ValueError, match="LEAKAGE"):
        validate_oof(a, "logistic", folds)
    a.loc[1,"REPDTE"] = a.loc[0,"REPDTE"]
    a.loc[1,"CERT"] = a.loc[0,"CERT"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_oof(a, "logistic", folds)


def test_cost_history_start_1984Q1_includes_jan_feb_and_excludes_1983(tmp_path, monkeypatch):
    raw = pd.DataFrame({
        "CERT": list(range(1,15)), "RESTYPE": ["FAILURE"] * 14,
        "FAILDATE": (["1983-12-31", "1984-01-15", "1984-02-28"] +
           [f"1999-01-{x:02d}" for x in range(1,12)]),
        "COST": list(range(1,15)),
    })
    fpath = tmp_path/"data_raw"/"fdic"/"failures.parquet"
    fpath.parent.mkdir(parents=True)
    fpath.write_bytes(b"synthetic fixture")
    def only_failures(path, *args, **kwargs):
        assert Path(path).name == "failures.parquet"
        return raw.copy()
    monkeypatch.setattr(pd, "read_parquet", only_failures)
    data = estimate_1984_2007_cost(tmp_path, .01)
    assert data["confirmed_failure_events_with_positive_cost"] == 13
    assert data["earliest_event"] == "1984-01-15"
    assert data["historical_cost_start"] == "1984-01-01"


def fixture_development(root: Path):
    out = root/"results"/"part5"
    private = root/"data_processed"/"part5"
    raw = root/"data_raw"/"fdic"
    out.mkdir(parents=True)
    private.mkdir(parents=True)
    raw.mkdir(parents=True)
    config = {"status":"DEVELOPMENT_LOCKED_TEST_UNTOUCHED",
       "chosen_model":"hist_gradient_boosting", "chosen_threshold":0.5,
       "training_feature_names":["equity_to_assets"],
       "costs":{"false_alarm_fraction_of_c_fn":.01,"c_fn_proxy_thousand_usd":1000}}
    (out/"development_selection.json").write_text(json.dumps(config))
    rows = []
    aud = []
    for i, q in enumerate(["19910331", "19980331", "20040331"], 1):
        aud.append({"fold":i,"val_start":q,"val_end":q,"val_rows":2,"val_positive_rows":1})
        rows.extend([{"fold":i,"model":"hist_gradient_boosting","CERT":10+i,
             "REPDTE":q,"y":0,"probability":.8},
             {"fold":i,"model":"hist_gradient_boosting","CERT":20+i,
             "REPDTE":q,"y":1,"probability":.7}])
    pd.DataFrame(aud).to_csv(out/"cv_fold_audit.csv",index=False)
    (private/"chosen_model_cv_predictions.parquet").write_bytes(b"synthetic fixture")
    (raw/"failures.parquet").write_bytes(b"synthetic fixture")
    failures = pd.DataFrame({"CERT":list(range(100,115)),
       "RESTYPE":["FAILURE"]*15,
       "FAILDATE":[f"1990-01-{i:02d}" for i in range(1,16)],
       "COST":[1000. + i for i in range(15)]})
    return pd.DataFrame(rows), failures


def test_recalibration_reads_only_development_and_is_write_once(tmp_path, monkeypatch):
    oof, failures = fixture_development(tmp_path)
    out = tmp_path/"results"/"part5"
    original_bytes = (out/"development_selection.json").read_bytes()
    def guarded_read(path, *args, **kwargs):
        name = Path(path).name
        assert name in {"chosen_model_cv_predictions.parquet", "failures.parquet"}, (
            "Test parquet read attempted", name)
        return oof.copy() if name.startswith("chosen") else failures.copy()
    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    result = recalibrate_development(tmp_path)
    assert result["test_data_accessed"] is False
    assert result["cv_retrained"] is False
    assert (out/"development_selection.json").read_bytes() == original_bytes
    cfg = json.loads((out/"development_selection_v3.json").read_text())
    assert cfg["threshold_method_version"] == METHOD_VERSION
    assert cfg["final_test_approved"] is False
    with pytest.raises(FileExistsError, match="already exist"):
        recalibrate_development(tmp_path)


def test_final_test_requires_explicit_approval_and_valid_v3(tmp_path):
    with pytest.raises(PermissionError, match="explicit user approval"):
        final_test_once(tmp_path, approved=False)
    with pytest.raises(FileNotFoundError, match="V3 development"):
        final_test_once(tmp_path, approved=True)


def test_existing_test_lock_blocks_both_v3_phases(tmp_path):
    out = tmp_path/"results"/"part5"; out.mkdir(parents=True)
    (out/"final_test_lock.json").write_text("{}")
    with pytest.raises(FileExistsError, match="held-out test"):
        recalibrate_development(tmp_path)
    with pytest.raises(FileExistsError, match="held-out test"):
        final_test_once(tmp_path, approved=True)


def test_legacy_final_test_is_disabled_even_if_confirmed(tmp_path):
    with pytest.raises(PermissionError, match="LEGACY FINAL TEST DISABLED"):
        disabled_legacy_final_test(tmp_path, confirm=True)


def test_final_test_lock_created_before_test_read_in_synthetic_fixture(tmp_path, monkeypatch):
    # Never exercise real test files: fake access and assert lock pre-exists.
    import src.part5_policy_v3 as mod
    oof, failures = fixture_development(tmp_path)
    def guarded_read(path, *args, **kwargs):
        return oof.copy() if Path(path).name.startswith("chosen") else failures.copy()
    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    mod.recalibrate_development(tmp_path)
    def mocked_require_inputs(root, include_test=False):
        assert include_test is False
        return ["equity_to_assets"], root/"data_processed"/"part4"
    def mocked_read_split(root, split, cols):
        if split == "test":
            assert (tmp_path/"results"/"part5"/"final_test_lock.json").exists()
        return pd.DataFrame({"CERT":[1,2,3,4],
             "REPDTE":["20070331"]*4 if split=="train" else ["20090331"]*4,
             "equity_to_assets":[.1,.2,.3,.4],
             "y_fail_next4q":[0,0,1,1]})
    class MockModel:
        def fit(self,x,y): return self
        def predict_proba(self,x):
            return np.c_[np.array([.95,.6,.8,.1]),np.array([.05,.4,.2,.9])]
    monkeypatch.setattr(mod, "require_inputs", mocked_require_inputs)
    monkeypatch.setattr(mod, "_read_split", mocked_read_split)
    monkeypatch.setattr(mod, "make_model", lambda name: MockModel())
    # Do not write actual parquet even in synthetic test: monkeypatch DataFrame method.
    monkeypatch.setattr(pd.DataFrame,"to_parquet",lambda *args,**kwargs: None)
    report=mod.final_test_once(tmp_path, approved=True)
    assert report["status"] == "V3_FINAL_TEST_COMPLETE_LOCKED_NO_REPEATS"
    assert (tmp_path/"results"/"part5"/"test_metrics_v3.json").is_file()
    with pytest.raises(FileExistsError, match="held-out test"):
        mod.final_test_once(tmp_path, approved=True)
