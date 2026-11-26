"""Focused Part 2 checks on synthetic bank histories."""
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.part2_panel import as_dates, clean_quarter, make_events, make_labels, quarter_list, run


class TestPart2(unittest.TestCase):
    # 1. Reporting dates and quarter range
    def test_dates(self):
        result = as_dates(pd.Series(["20230331", "2023-06-30", "9999-12-31", "9999/12/31", "99999999", "12/31/9999"]))
        self.assertEqual(result.iloc[0], pd.Timestamp("2023-03-31"))
        self.assertEqual(result.iloc[1], pd.Timestamp("2023-06-30"))
        self.assertTrue(result.iloc[2:].isna().all())
        self.assertEqual(quarter_list(date(2023, 3, 31), date(2023, 9, 30)),
                         ["20230331", "20230630", "20230930"])

    def events(self):
        institutions = pd.DataFrame({"CERT": [1, 2, 3], "ACTIVE": [0, 0, 1],
                                     "CHANGEC1": [211, 223, None],
                                     "ENDEFYMD": ["2023-08-10", "2023-07-01", "9999-12-31"]})
        failures = pd.DataFrame({"CERT": [1], "FAILDATE": ["2023-08-10"],
                                      "RESTYPE": ["FAILURE"]})
        return make_events(institutions, failures)

    # 2. A failure is not a merger; active banks are right-censored at extract date
    def test_events(self):
        kinds = self.events().set_index("CERT")["eventual_outcome"].to_dict()
        self.assertEqual(kinds, {1: "failure", 2: "merger_candidate", 3: "unresolved"})

    # 3. Label excludes same-day failure and censors mergers in the risk window
    def test_labels(self):
        cert = pd.Series([1, 2, 3])
        y, exclusions = make_labels(cert, "20230630", self.events())
        self.assertEqual(y.iloc[0], 1)
        self.assertTrue(pd.isna(y.iloc[1]))  # Merger before four-quarter horizon
        self.assertEqual(y.iloc[2], 0)
        self.assertEqual(exclusions["competing_exit"], 1)
        y_late, _ = make_labels(cert, "20250930", self.events())
        self.assertTrue(y_late.isna().all())
        same_day = self.events().copy()
        same_day.loc[same_day.CERT.eq(1), "failure_date"] = pd.Timestamp("2023-06-30")
        same_day.loc[same_day.CERT.eq(1), "event_date"] = pd.Timestamp("2023-06-30")
        y_same_day, _ = make_labels(pd.Series([1]), "20230630", same_day)
        self.assertTrue(y_same_day.isna().all())

    # 4. Distinguish assistance, repeated failures and missing historical CERT
    def test_reconcile_failure_events(self):
        inst = pd.DataFrame({"CERT": [1, 2, 3, 4], "ACTIVE": [0, 0, 0, 1],
                             "CHANGEC1": [None] * 4,
                             "ENDEFYMD": ["9999-12-31"] * 4})
        fails = pd.DataFrame({
            "CERT": [None, 1, 1, 2, 2, 3, 3, 4],
            "FAILDATE": ["1970-01-01", "1988-04-20", "1992-10-30",
                         "1985-05-10", "1987-12-31", "1986-03-27",
                         "1989-02-17", "2008-01-25"],
            "RESTYPE": ["FAILURE", "ASSISTANCE", "FAILURE", "ASSISTANCE",
                        "ASSISTANCE", "FAILURE", "FAILURE", "FAILURE"]})
        events = make_events(inst.loc[inst.CERT.ne(3)], fails).set_index("CERT")
        self.assertEqual(events.loc[1, "failure_date"], pd.Timestamp("1992-10-30"))
        self.assertTrue(pd.isna(events.loc[2, "failure_date"]))
        self.assertEqual(events.loc[3, "eventual_outcome"], "ambiguous_failure")
        y, counts = make_labels(pd.Series([3, 4]), "20070630", events.reset_index())
        self.assertTrue(pd.isna(y.iloc[0]))
        self.assertEqual(y.iloc[1], 1)
        self.assertEqual(counts["ambiguous_failure"], 1)

    # 5. Exclude invalid assets with an explicit audit, never silently impute
    def test_clean(self):
        raw = pd.DataFrame({"CERT": [1, 2, 3], "REPDTE": ["20230331"] * 3,
                            "ASSET": [1000, None, 0], "DEP": [800, -2, 1],
                            "EQ": [100, 0, 0], "ROA": [1.2, None, None]})
        clean, metrics, flags = clean_quarter(raw, "20230331")
        self.assertEqual(len(clean), 1)
        self.assertEqual(flags["excluded_asset_missing"], 1)
        self.assertEqual(flags["excluded_asset_nonpositive"], 1)
        self.assertEqual(flags["negative_deposits"], 1)
        self.assertEqual(next(item for item in metrics if item["field"] == "ASSET")["missing"], 1)
        with self.assertRaises(ValueError):
            clean_quarter(pd.concat([raw, raw.iloc[[0]]], ignore_index=True), "20230331")

    # 6. Three-quarter integration, including a missing-bank-quarter gap
    def test_integrated_output(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "raw"
            (src / "financials").mkdir(parents=True)
            quarters = ["20230331", "20230630", "20230930"]
            for q, certs in zip(quarters, [[1, 2, 3], [1, 3], [1, 2, 3]]):
                pd.DataFrame({"CERT": certs, "REPDTE": [q] * len(certs),
                              "ASSET": [1000] * len(certs), "DEP": [700] * len(certs),
                              "EQ": [100] * len(certs), "ROA": [1.0] * len(certs)}).to_pickle(
                                  src / "financials" / f"REPDTE={q}.parquet")
            pd.DataFrame({"CERT": [1, 2, 3], "ACTIVE": [0, 0, 1],
                          "CHANGEC1": [211, 223, None],
                          "ENDEFYMD": ["2023-08-10", "2023-07-01", "9999-12-31"]}).to_pickle(
                              src / "institutions.parquet")
            pd.DataFrame({"CERT": [1], "FAILDATE": ["2023-08-10"],
                                      "RESTYPE": ["FAILURE"]}).to_pickle(
                src / "failures.parquet")
            with patch("src.part2_panel.pd.read_parquet", side_effect=pd.read_pickle), \
                 patch.object(pd.DataFrame, "to_parquet", lambda df, path, **kw: df.to_pickle(path)):
                outcome = run(src, root / "processed", root / "results",
                              start=date(2023, 3, 31), end=date(2023, 9, 30),
                              label_end=date(2023, 9, 30))
            self.assertEqual(outcome["raw_rows"], 8)
            self.assertEqual(outcome["panel_rows"], 8)
            self.assertEqual(outcome["reporting_gaps"], 1)
            states = pd.read_pickle(root / "processed" / "event_lookup_AUDIT_ONLY.parquet")
            self.assertEqual(states.set_index("CERT").loc[3, "eventual_outcome"], "right_censored")
            panel = pd.read_pickle(root / "processed" / "panel/REPDTE=20230331.parquet")
            self.assertFalse({"FAILDATE", "CHANGEC1", "ACTIVE", "y_fail_next4q"} & set(panel.columns))
            missing = pd.read_csv(root / "results" / "missing_by_decade.csv")
            self.assertIn("missing_pct", missing)
            self.assertTrue((root / "results" / "reporting_gaps.csv").exists())


if __name__ == "__main__":
    unittest.main()
