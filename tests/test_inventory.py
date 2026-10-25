"""Synthetic regression checks for calibration timing and inventory arithmetic."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from inventory import calibrate_one, inventory_outcome, block_mean_interval, QUANTILES


class InventoryTests(unittest.TestCase):
    def test_calibration_boundary_fallback_and_monotonicity(self):
        dates = pd.date_range("2010-01-04", periods=12, freq="W-MON")
        historical = pd.DataFrame({"variant": "original", "method": "mean_4_weeks", "StockCode": "12345",
                                   "origin": dates, "target_end": dates + pd.Timedelta(days=27),
                                   "residual": np.arange(12) - 5., "residual_scale": 4.,
                                   "forecast_id": [f"past-{i}" for i in range(12)]})
        row = pd.Series({"variant": "original", "method": "mean_4_weeks", "StockCode": "12345",
                         "origin": pd.Timestamp("2011-01-03"), "prediction": 10., "actual": 20.,
                         "residual_scale": 8., "forecast_id": "current"})
        estimate, members = calibrate_one(row, historical)
        self.assertAlmostEqual(estimate["q80"], 13.8)
        self.assertEqual(estimate["calibration_source"], "product")
        # A window ending exactly at the new forecast's date is still unavailable.
        pending = historical.iloc[:1].copy()
        pending["target_end"] = row.origin
        pending["residual"] = 1e9
        pending["forecast_id"] = "unavailable"
        altered = pd.concat([historical, pending], ignore_index=True)
        changed_row = row.copy()
        changed_row["actual"] = 1e9
        changed, changed_members = calibrate_one(changed_row, altered)
        self.assertEqual(estimate, changed)
        self.assertEqual(set(members.calibration_forecast_id), set(changed_members.calibration_forecast_id))
        unseen = row.copy()
        unseen["StockCode"] = "99999"
        pooled, _ = calibrate_one(unseen, historical)
        self.assertEqual(pooled["calibration_source"], "pooled_normalized")
        self.assertAlmostEqual(pooled["q80"], 17.6)
        clipped, _ = calibrate_one(row, historical.assign(residual=-100.))
        self.assertTrue(all(clipped[key] == 0 for key in QUANTILES))
        with self.assertRaises(ValueError):
            calibrate_one(unseen, historical.head(3))

    def test_orders_costs_and_paired_block_arithmetic(self):
        result = inventory_outcome([20, 12], [15.2, 15.2], [3, 20], 4, 1)
        np.testing.assert_array_equal(result["order"], [13, 0])
        np.testing.assert_array_equal(result["available"], [16, 20])
        np.testing.assert_array_equal(result["shortage"], [4, 0])
        np.testing.assert_array_equal(result["surplus"], [0, 8])
        np.testing.assert_array_equal(result["cost"], [16, 8])
        # Future actuals affect evaluation cost, never the chosen order.
        changed = inventory_outcome([2000, 1200], [15.2, 15.2], [3, 20], 4, 1)
        np.testing.assert_array_equal(result["order"], changed["order"])
        np.testing.assert_array_equal(block_mean_interval(np.repeat(-5., 10)), [-5., -5.])
        for args in [(1, -1, 0, 4, 1), (1, 1, -1, 4, 1), (1, 1, 0, -1, 1), (1, 1, 0, 0, 0), (np.nan, 1, 0, 4, 1)]:
            with self.assertRaises(ValueError):
                inventory_outcome(*args)


if __name__ == "__main__":
    unittest.main()
