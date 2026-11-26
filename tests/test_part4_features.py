"""Synthetic unit tests; NO FDIC downloads, no user data, no Parquet dependency."""
import unittest
import numpy as np
import pandas as pd
from src.part4_features import (
    FEATURE_COLUMNS, RAW_FIELDS, LABEL, SOURCE_DATE, build_lagged_one_quarter,
    chronological_bucket, make_financial_features, make_preprocessor,
    previous_quarter, validate_lagged_alignment,
)


def prior_panel(q="20081231"):
    base = {f: [1.0, 2.0] for f in RAW_FIELDS}
    base.update({
        "CERT": [11, 22], "REPDTE": [q, q],
        "ASSET": [100., 80.], "EQ": [10., 8.], "LNLSGR": [0., 40.],
        "NCLNLS": [5., 4.], "P9ASSET": [1., 1.], "NAASSET": [2., 3.],
        "NTLNLS": [-1., 2.], "LNATRES": [4., 3.],
        "BRO": [20., 10.], "DEP": [60., 70.],
    })
    return pd.DataFrame(base)


class Part4Tests(unittest.TestCase):
    def test_lag_dates_and_split(self):
        self.assertEqual(previous_quarter("20090331"), "20081231")
        self.assertEqual(previous_quarter("20000331"), "19991231")
        with self.assertRaises(ValueError):
            previous_quarter("20000101")
        self.assertEqual(chronological_bucket("20071231"), "train")
        self.assertEqual(chronological_bucket("20080331"), "gap")
        self.assertEqual(chronological_bucket("20081231"), "gap")
        self.assertEqual(chronological_bucket("20090331"), "test")
        self.assertEqual(chronological_bucket("20250630"), "test")
        self.assertEqual(chronological_bucket("20250930"), "outside")

    def test_denominator_invalid_stays_missing(self):
        f = make_financial_features(prior_panel())
        self.assertTrue(np.isnan(f.loc[0, "noncurrent_loans_to_loans"]))
        self.assertAlmostEqual(f.loc[1, "noncurrent_loans_to_loans"], 0.1)
        self.assertAlmostEqual(f.loc[0, "equity_to_assets"], 0.10)
        self.assertTrue(np.isnan(f.loc[0, "net_chargeoffs_to_loans"]))
        self.assertEqual(set(FEATURE_COLUMNS), set(f.columns) - {"CERT", "REPDTE"})

    def test_label_join_correct_bank_and_lag(self):
        labels = pd.DataFrame({"CERT": [22, 11, 99], "REPDTE": ["20090331"]*3, LABEL: [0, 1, 1]})
        out, excluded = build_lagged_one_quarter(labels, prior_panel(), q="20090331", prior_q="20081231")
        self.assertEqual(excluded, 1)
        self.assertEqual(len(out), 2)
        self.assertEqual(out.set_index("CERT").loc[11, LABEL], 1)
        self.assertEqual(out.set_index("CERT").loc[22, LABEL], 0)
        self.assertEqual(out.set_index("CERT").loc[11, SOURCE_DATE], "20081231")
        validate_lagged_alignment(out)

    def test_refuses_same_or_future_quarter(self):
        labels = pd.DataFrame({"CERT": [11], "REPDTE": ["20090331"], LABEL: [1]})
        with self.assertRaises(ValueError):
            build_lagged_one_quarter(labels, prior_panel("20090331"), q="20090331", prior_q="20090331")
        future = pd.DataFrame({"CERT": [11], "REPDTE": ["20090331"], SOURCE_DATE: ["20090630"],
                               LABEL: [1], **{f: [1.0] for f in FEATURE_COLUMNS}})
        with self.assertRaises(ValueError):
            validate_lagged_alignment(future)

    def test_refuses_leak_column(self):
        bad = prior_panel().assign(FAILDATE=["2009-04-01", None])
        with self.assertRaises(ValueError):
            make_financial_features(bad)

    def test_train_only_preprocessor_medians(self):
        train = pd.DataFrame({"a": [1., np.nan, 3.], "b": [10., 20., 30.]})
        test = pd.DataFrame({"a": [np.nan, 10000.], "b": [100000., np.nan]})
        pp = make_preprocessor().fit(train)
        self.assertAlmostEqual(float(pp.named_steps["imputer"].statistics_[0]), 2.)
        before = pp.named_steps["imputer"].statistics_.copy()
        transformed = pp.transform(test)
        np.testing.assert_array_equal(before, pp.named_steps["imputer"].statistics_)
        self.assertTrue(np.isfinite(transformed).all())


if __name__ == "__main__":
    unittest.main()
