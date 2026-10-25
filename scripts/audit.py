"""Verify local source exports and build an audited weekly recorded-sales panel."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(root):
    manifest = json.loads((root / "data/csv-manifest.json").read_text())
    frames = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Source bytes differ from manifest: {path.name}")
        frame = pd.read_csv(path, dtype={name: "string" for name in COLUMNS if name not in ["Quantity", "Price"]})
        if list(frame.columns) != COLUMNS or list(frame.columns) != entry["columns"] or len(frame) != entry["rows"]:
            raise ValueError(f"Source schema or row count differs from manifest: {path.name}")
        if frame[["Invoice", "StockCode", "InvoiceDate", "Country"]].isna().any().any():
            raise ValueError("Missing transaction keys")
        if not np.isfinite(frame[["Quantity", "Price"]].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite quantity or price")
        if not frame.Quantity.mod(1).eq(0).all():
            raise ValueError("Nonintegral quantities")
        frame["timestamp"] = pd.to_datetime(frame.InvoiceDate, errors="raise")
        frame["source_file"] = path.name
        frame["source_row"] = np.arange(len(frame))  # zero-based CSV data row, excluding header
        frame["sheet"] = int(path.stem.split("_")[1].removeprefix("sheet"))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), manifest


def reconcile_sheets(frame):
    """Later workbook sheet owns every date it shares with an earlier sheet."""
    frame = frame.copy()
    frame["date"] = frame.timestamp.dt.normalize()
    owner = frame.groupby("date").sheet.transform("max")
    return frame.loc[frame.sheet.eq(owner)].copy(), frame.loc[frame.sheet.ne(owner)].copy()


def sales_mask(frame, config):
    # Product codes are a documented heuristic; exclusions are reported for review.
    return (frame.Country.eq(config["country"])
            & ~frame.Invoice.str.upper().str.startswith("C")
            & frame.Quantity.gt(0) & frame.Price.gt(0)
            & frame.StockCode.str.fullmatch(config["product_code_pattern"]))


def complete_bounds(frame):
    first = frame.timestamp.min().normalize()
    last = frame.timestamp.max().normalize()
    start = first + pd.Timedelta(days=(7 - first.weekday()) % 7)
    # The final recorded date is conservatively treated as incomplete, even on Sunday.
    end = last - pd.Timedelta(days=last.weekday() + 1)
    return start, end


def make_panel(frame, config):
    start, end = complete_bounds(frame)
    sales = frame.loc[sales_mask(frame, config)].copy()
    sales["week"] = sales.timestamp.dt.to_period("W-SUN").dt.start_time
    sales = sales.loc[sales.week.ge(start) & sales.week.le(end - pd.Timedelta(days=6))]
    weekly = sales.groupby(["StockCode", "week"]).Quantity.sum().rename("units")
    development = pd.Timestamp(config["development_start"])
    test = pd.Timestamp(config["test_start"])
    if not (start < development < test <= end) or development.weekday() or test.weekday():
        raise ValueError("Split boundaries must be ordered Mondays inside complete history")
    training = weekly.reset_index().loc[lambda x: x.week.lt(development)]
    candidates = training.groupby("StockCode").agg(
        first_training_week=("week", "min"), active_training_weeks=("week", "size"),
        training_units=("units", "sum"))
    candidates["history_weeks"] = ((development - candidates.first_training_week).dt.days // 7)
    selected = candidates.loc[
        candidates.history_weeks.ge(config["minimum_history_weeks"])
        & candidates.active_training_weeks.ge(config["minimum_active_training_weeks"])
    ].reset_index().sort_values(["training_units", "StockCode"], ascending=[False, True]).head(config["maximum_products"])
    if selected.empty:
        raise ValueError("No products satisfy training-only eligibility")
    panels = []
    for row in selected.itertuples(index=False):
        dates = pd.date_range(row.first_training_week, end - pd.Timedelta(days=6), freq="W-MON")
        units = weekly.loc[row.StockCode].reindex(dates, fill_value=0)
        panels.append(pd.DataFrame({"StockCode": row.StockCode, "week": dates, "units": units.to_numpy()}))
    panel = pd.concat(panels, ignore_index=True).sort_values(["StockCode", "week"]).reset_index(drop=True)
    panel["split"] = np.select([panel.week.lt(development), panel.week.lt(test)], ["train", "development"], default="test")
    # Never create labels crossing a partition boundary. Dates denote target-week starts.
    origins = []
    for split in ["development", "test"]:
        dates = sorted(panel.loc[panel.split.eq(split), "week"].unique())
        for origin in dates:
            last_target = origin + pd.Timedelta(weeks=config["horizon_weeks"] - 1)
            if last_target <= dates[-1]:
                origins.append({"split": split, "origin": origin,
                                "history_through": origin - pd.Timedelta(days=1),
                                "target_end": last_target + pd.Timedelta(days=6),
                                "horizon_weeks": config["horizon_weeks"]})
    return sales, panel, selected, pd.DataFrame(origins), start, end


def summarize(raw, clean, removed, sales, panel, selected, origins, start, end, config):
    training_sales = sales.loc[sales.week.lt(pd.Timestamp(config["development_start"]))]
    selected_sales = training_sales.loc[training_sales.StockCode.isin(selected.StockCode)]
    repeats = clean.duplicated(COLUMNS)
    repeated_sales = sales.loc[sales.duplicated(COLUMNS)]
    outside = clean.loc[~clean.StockCode.str.fullmatch(config["product_code_pattern"])]
    return {
        "raw_rows": len(raw), "reconciled_rows": len(clean),
        "missing_by_column": {k: int(v) for k, v in raw[COLUMNS].isna().sum().items()},
        "first_timestamp": str(raw.timestamp.min()), "last_timestamp": str(raw.timestamp.max()),
        "sheets": {str(k): {"rows": len(g), "start": str(g.timestamp.min()), "end": str(g.timestamp.max()),
                             "exact_repeats_beyond_first": int(g.duplicated(COLUMNS).sum())} for k, g in raw.groupby("sheet")},
        "overlap": {"removed_earlier_sheet_rows": len(removed),
                    "dates": [str(v.date()) for v in sorted(removed.date.unique())],
                    "removed_rows_with_exact_later_sheet_match": int(pd.MultiIndex.from_frame(removed[COLUMNS]).isin(pd.MultiIndex.from_frame(clean.loc[clean.date.isin(removed.date.unique()), COLUMNS])).sum())},
        "raw_exact_repeats_beyond_first": int(raw.duplicated(COLUMNS).sum()),
        "retained_exact_repeats_beyond_first": int(repeats.sum()),
        "reconciled_cancellations": int(clean.Invoice.str.upper().str.startswith("C").sum()),
        "reconciled_nonpositive_quantity": int(clean.Quantity.le(0).sum()),
        "reconciled_nonpositive_price": int(clean.Price.le(0).sum()),
        "excluded_code_rows": {str(k): int(v) for k, v in outside.StockCode.value_counts().items()},
        "countries": {str(k): int(v) for k, v in clean.Country.value_counts().items()},
        "complete_weeks_start": str(start.date()), "complete_weeks_end": str(end.date()),
        "filtered_sales_rows_complete_weeks": len(sales), "filtered_sales_units": int(sales.Quantity.sum()),
        "sales_missing_customer_id": int(sales["Customer ID"].isna().sum()),
        "sales_repeat_units_sensitivity": int(repeated_sales.Quantity.sum()),
        "selected_products": len(selected), "panel_rows": len(panel),
        "panel_zero_fraction": float(panel.units.eq(0).mean()),
        "selected_training_unit_share": float(selected_sales.Quantity.sum() / training_sales.Quantity.sum()),
        "panel_rows_by_split": {str(k): int(v) for k, v in panel.split.value_counts().items()},
        "origins_by_split": {str(k): int(v) for k, v in origins.split.value_counts().items()},
        "selected_product_training_statistics": json.loads(selected.to_json(orient="records", date_format="iso")),
    }


def report(r):
    return f"""# Recorded-sales data audit

Generated by `python scripts/audit.py`; descriptive audit, no trained model or forecast evaluation.

- Verified all 12 CSV checksums, byte sizes, schemas and row counts: **{r['raw_rows']:,} raw rows**.
- Coverage: {r['first_timestamp']} to {r['last_timestamp']}; complete calendar weeks retained: {r['complete_weeks_start']} through {r['complete_weeks_end']}.
- Missing customer IDs: {r['missing_by_column']['Customer ID']:,}; missing descriptions: {r['missing_by_column']['Description']:,}. Neither field is needed for aggregate sales; missing customer IDs are retained.

## Sheet overlap and repeated rows

The sheets share {len(r['overlap']['dates'])} calendar dates in December 2010. Use the later sheet on shared dates: remove {r['overlap']['removed_earlier_sheet_rows']:,} earlier-sheet rows; {r['overlap']['removed_rows_with_exact_later_sheet_match']:,} have an exact match in the retained sheet. This calendar ownership rule preserves later-sheet line multiplicity and prevents workbook overlap from doubling sales.

There are {r['raw_exact_repeats_beyond_first']:,} exact repeated raw rows beyond first. After sheet reconciliation, {r['retained_exact_repeats_beyond_first']:,} remain. Retain these within-sheet repeats: no line-item key establishes that they are errors. Dropping exact repeats from eligible UK sales would remove {r['sales_repeat_units_sensitivity']:,} units across complete weeks. This is a descriptive sensitivity, not a preferred correction; compare policies both ways before promotion.

## Target construction

After reconciliation: {r['reconciled_rows']:,} rows, {r['reconciled_cancellations']:,} cancellation-invoice rows, {r['reconciled_nonpositive_quantity']:,} nonpositive quantities and {r['reconciled_nonpositive_price']:,} nonpositive prices. These counts overlap and are not an additive exclusion waterfall. A separate exclusion table in JSON records nonmatching stock codes, including postage and adjustments.

Keep UK rows with positive quantity, positive price, noncancellation invoice and stock code matching the configured five-digit product pattern. The pattern is a scope heuristic, not a verified product catalog. Exclude boundary calendar weeks. This leaves {r['filtered_sales_rows_complete_weeks']:,} lines and {r['filtered_sales_units']:,} units. Do not net cancellations into positive sales or assume they occurred in the original sale week. Extreme positive orders are retained for later sensitivity analysis.

Choose the top {r['selected_products']} products by **training-only** units after requiring 26 history weeks and 13 active training weeks. They cover {r['selected_training_unit_share']:.2%} of filtered training units. Do not generalize to new products or the catalog. The resulting {r['panel_rows']:,} product-weeks have {r['panel_zero_fraction']:.2%} zeros. Zero-fill only from each product's first observed complete-week sale; earlier weeks do not exist in the panel. Zeros mean no retained recorded sale, not verified in-stock zero demand. Closure, delisting and stockout status are unknown, including internal dates without transactions.

## Temporal evaluation readiness

Panel row counts: {r['panel_rows_by_split']}. Four-week rolling origins: {r['origins_by_split']}. Training ends before 2011-06-06; development target weeks precede 2011-09-05; final target weeks start on that date. Target windows never cross partitions. Overlapping origins require origin-aware uncertainty, not an independent product-row test. No candidate fitting, interval calibration or final-test prediction has run.

Daily transaction timestamps are treated as local recorded wall time; no timezone is supplied. Country and SKU define the series; invoice/customer/description and contemporaneous target-week price are not model features. No inventory, supplier lead time, purchase cost, stock availability, lost-sales label or replenishment action is present. Forecast **recorded positive unit sales**, then evaluate hypothetical inventory costs with stated assumptions. See [charter](../project-charter.md) and [evaluation plan](../evaluation-plan.md).
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    root = parser.parse_args().root
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    raw, manifest = load_source(root)
    clean, removed = reconcile_sheets(raw)
    sales, panel, selected, origins, start, end = make_panel(clean, config)
    result = summarize(raw, clean, removed, sales, panel, selected, origins, start, end, config)
    result["config"] = config
    result["runtime"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    result["lineage"] = {"sources": {x["path"]: x["sha256"] for x in manifest["files"]},
                         "manifest_sha256": sha256(root / "data/csv-manifest.json"),
                         "config_sha256": sha256(config_path), "audit_code_sha256": sha256(Path(__file__))}
    interim, processed, reports = [root / p for p in ["data/interim", "data/processed", "docs/reports"]]
    for directory in [interim, processed, reports]:
        directory.mkdir(parents=True, exist_ok=True)
    products = {interim / "uk-sales.csv": sales[["source_file", "source_row", "StockCode", "timestamp", "week", "Quantity", "Price"]],
                processed / "weekly-sales.csv": panel, processed / "selected-products.csv": selected,
                processed / "forecast-origins.csv": origins}
    result["generated_sha256"] = {}
    for path, frame in products.items():
        frame.to_csv(path, index=False)
        result["generated_sha256"][str(path.relative_to(root))] = sha256(path)
    (reports / "data-audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (reports / "01-data-audit.md").write_text(report(result))
    print(json.dumps({k: result[k] for k in ["raw_rows", "reconciled_rows", "selected_products", "panel_rows", "origins_by_split"]}, indent=2))


if __name__ == "__main__":
    main()
