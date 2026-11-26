"""Small reproducible tests; no data download, benchmarks or Spark startup."""
import unittest

import pandas as pd

from src.part3_processing import aggregate_pandas, compare_outputs


class TestPart3(unittest.TestCase):
    def setUp(self):
        self.fin = pd.DataFrame({
            "CERT": [1, 2, 1, 3, 4],
            "REPDTE": ["20240331", "20240331", "20240630", "20240331", "20260930"],
            "STNAME": ["NY", "NY", "NY", None, "CA"],
            "ASSET": [100.0, 200.0, 110.0, 50.0, 100.0],
            "DEP": [80.0, None, 90.0, 40.0, 10.0],
        })
        self.lab = pd.DataFrame({
            "CERT": [1, 2, 1, 3, 4],
            "REPDTE": ["20240331", "20240331", "20240630", "20240331", "20260930"],
            "y_fail_next4q": pd.Series([1, 0, 1, pd.NA, pd.NA], dtype="Int8"),
        })

    def test_pandas_logic(self):
        result = aggregate_pandas(self.fin, self.lab)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "bank_quarters"], 3)
        self.assertEqual(result.loc[0, "distinct_banks"], 2)
        self.assertEqual(result.loc[0, "positive_labels"], 2)
        self.assertEqual(result.loc[0, "deposit_reported_rows"], 2)
        self.assertEqual(result.loc[0, "total_assets_usd000"], 410)
        self.assertEqual(result.loc[0, "reported_deposits_usd000"], 170)

    def test_parity_detects_error(self):
        correct = aggregate_pandas(self.fin, self.lab)
        self.assertTrue(compare_outputs(correct, correct)["passed"])
        changed = correct.copy()
        changed.loc[0, "positive_labels"] += 1
        with self.assertRaises(AssertionError):
            compare_outputs(correct, changed)

    @unittest.skipUnless(__import__("importlib").util.find_spec("polars"), "Polars unavailable")
    def test_polars_same_logic(self):
        import polars as pl
        from src.part3_processing import aggregate_polars

        fin = pl.from_pandas(self.fin)
        lab = pl.from_pandas(self.lab)
        result = aggregate_polars(fin.lazy(), lab.lazy()).collect().to_pandas()
        self.assertTrue(compare_outputs(aggregate_pandas(self.fin, self.lab), result)["passed"])


if __name__ == "__main__":
    unittest.main()
