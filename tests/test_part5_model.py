import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.part5_model import (
    choose_model, cost_sweep, make_model, quarter_folds, score_metrics,
    summarize_bank_policy, _parse_costs,
)


def test_quarter_folds_purge_four_whole_quarters():
    quarters = pd.period_range("1984Q2", "2007Q4", freq="Q")
    dates = np.repeat(quarters.to_timestamp(how="end").strftime("%Y%m%d"), 5)
    folds = list(quarter_folds(dates, n_splits=3, gap=4))
    assert len(folds) == 3
    for _, tr, val in folds:
        assert set(tr).isdisjoint(set(val))
        assert pd.Period(pd.to_datetime(tr[-1], format="%Y%m%d"), freq="Q") + 4 < pd.Period(
            pd.to_datetime(val[0], format="%Y%m%d"), freq="Q")
        assert all(s in tr for s in np.unique(dates[np.isin(dates, tr)]))


def test_four_row_gap_is_not_quarter_gap():
    # With repeated banks per quarter, the quarter splitter still excludes whole quarters.
    quarter_list = pd.period_range("1990Q1", periods=60, freq="Q")
    dates = np.repeat(quarter_list.to_timestamp(how="end").strftime("%Y%m%d"), 100)
    for _, tr, val in quarter_folds(dates):
        assert len(set(tr).intersection(val)) == 0
        assert all(np.sum(dates == q) == 100 for q in tr)


def test_preprocessor_is_new_and_median_train_only():
    a, b = make_model("logistic"), make_model("logistic")
    assert a is not b
    assert isinstance(a, Pipeline)
    assert a.named_steps["preprocessor"] is not b.named_steps["preprocessor"]
    x = pd.DataFrame({"a": [0.0, 10.0, np.nan, 20.0, 30.0, 40.0],
                      "b": [5., 6., 7., 8., 9., 10.]})
    y = [0, 1, 0, 1, 0, 1]
    a.fit(x, y)
    median = a.named_steps["preprocessor"].named_steps["imputer"].statistics_[0]
    assert median == 20
    _ = a.predict_proba(pd.DataFrame({"a": [99999.], "b": [12345.]}))
    assert a.named_steps["preprocessor"].named_steps["imputer"].statistics_[0] == 20


def test_cost_history_excludes_assistance_future_and_ambiguous():
    data = pd.DataFrame({
        "CERT": list(range(1, 13)) + [1, 30, 30, 50],
        "RESTYPE": ["FAILURE"] * 12 + ["ASSISTANCE", "FAILURE", "FAILURE", "FAILURE"],
        "FAILDATE": [f"2001-01-{i:02d}" for i in range(1, 13)]
                    + ["2001-02-01", "2002-02-01", "2004-02-01", "2009-01-01"],
        "COST": [float(i) for i in range(1, 13)] + [1000., 500., 500., 900.],
    })
    actual = _parse_costs(data)
    assert actual.CERT.nunique() == 12
    assert actual.cost_usd_thousands.median() == 6.5
    assert 30 not in actual.CERT.values and 50 not in actual.CERT.values


def test_bank_policy_counts_each_bank_once_per_fold():
    scores = pd.DataFrame({"fold": [1, 1, 1, 2], "CERT": [10, 10, 20, 10],
                           "y": [0, 1, 0, 0], "probability": [0.2, 0.8, 0.4, 0.1]})
    policy = summarize_bank_policy(scores)
    assert len(policy) == 3
    row = policy[(policy.fold == 1) & (policy.CERT == 10)].iloc[0]
    assert row.y == 1 and row.probability == 0.8


def test_threshold_comes_from_oof_cost_not_test():
    policy = pd.DataFrame({"fold": [1]*6, "CERT": list(range(6)),
                           "y": [0, 0, 0, 1, 1, 1],
                           "probability": [0.10, 0.15, 0.30, 0.4, 0.7, 0.95]})
    t, table = cost_sweep(policy, c_fn=1000., preferred_fraction=0.01)
    assert 0.30 < t <= 0.4
    assert set(table.false_alarm_fraction) == {0.005, 0.01, 0.05}
    assert not table.empty


def test_ap_confusion_metrics():
    r = score_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.9, 0.3]), 0.5)
    assert (r["tn"], r["fp"], r["fn"], r["tp"]) == (1, 1, 1, 1)
    assert r["average_precision"] > 0.5
    assert r["precision"] == 0.5


def test_model_choice_deterministic_tie():
    cv = pd.DataFrame({"model": ["logistic", "hist_gradient_boosting"] * 3,
                       "average_precision": [0.2] * 6})
    assert choose_model(cv)[0] == "logistic"


def test_final_test_refuses_unapproved_call(tmp_path):
    from src.part5_model import final_test
    with pytest.raises(PermissionError):
        final_test(tmp_path, confirm=False)
