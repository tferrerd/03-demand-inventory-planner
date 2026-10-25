"""Synthetic boundary/regression checks; not additional training observations."""

import importlib.util
import json
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit", ROOT / "scripts/audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_overlap_multiplicity_and_sales_filters(self):
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(["2010-12-01"] * 4 + ["2010-11-30"]),
            "sheet": [1, 2, 2, 1, 1], "Quantity": [1, 1, 1, 4, 9],
        })
        clean, removed = audit.reconcile_sheets(frame)
        self.assertEqual(clean.Quantity.tolist(), [1, 1, 9])
        self.assertEqual(len(removed), 2)
        sales = pd.DataFrame({"Country": ["United Kingdom"] * 7 + ["France"],
                              "Invoice": ["1", "C1", "1", "1", "1", "1", "1", "1"],
                              "Quantity": [1, 1, -1, 1, 1, 1, 1, 1],
                              "Price": [1, 1, 1, 0, 1, -1, 1, 1],
                              "StockCode": ["12345", "12345", "12345", "12345", "POST", "12345", "12345a", "12345"]})
        config = json.loads((ROOT / "config.json").read_text())
        self.assertEqual(audit.sales_mask(sales, config).tolist(), [True, False, False, False, False, False, True, False])

    def test_panel_selection_ignores_future_and_origins_stay_in_split(self):
        config = json.loads((ROOT / "config.json").read_text())
        dates = pd.date_range("2009-12-07", "2011-12-05", freq="W-MON")
        frame = pd.DataFrame({"timestamp": dates, "Quantity": 2, "StockCode": "12345",
                              "Country": "United Kingdom", "Invoice": "1", "Price": 1})
        # An interior zero week must be filled; a later product must have no pre-birth rows.
        frame = frame.loc[frame.timestamp.ne(pd.Timestamp("2010-06-07"))]
        later = frame.loc[frame.timestamp.ge(pd.Timestamp("2010-06-14"))].assign(StockCode="23456")
        future = frame.loc[frame.timestamp.ge(pd.Timestamp(config["test_start"]))].assign(StockCode="99999", Quantity=1000000)
        full = pd.concat([frame, later, future], ignore_index=True)
        _, panel, selected, origins, _, end = audit.make_panel(full, config)
        self.assertEqual(set(selected.StockCode), {"12345", "23456"})
        self.assertEqual(panel.loc[panel.StockCode.eq("23456"), "week"].min(), pd.Timestamp("2010-06-14"))
        self.assertEqual(panel.loc[panel.StockCode.eq("12345") & panel.week.eq(pd.Timestamp("2010-06-07")), "units"].item(), 0)
        self.assertEqual(str(end.date()), "2011-12-04")
        self.assertEqual(origins.groupby("split").size().to_dict(), {"development": 10, "test": 10})
        self.assertTrue((origins.history_through < origins.origin).all())
        dev = origins.loc[origins.split.eq("development")]
        self.assertTrue((dev.target_end < pd.Timestamp(config["test_start"])).all())
        changed = full.copy()
        changed.loc[changed.timestamp.ge(pd.Timestamp(config["development_start"])), "Quantity"] *= 123
        _, _, changed_selection, changed_origins, _, _ = audit.make_panel(changed, config)
        pd.testing.assert_frame_equal(selected, changed_selection)
        pd.testing.assert_frame_equal(origins, changed_origins)


if __name__ == "__main__":
    unittest.main()
