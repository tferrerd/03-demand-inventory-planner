# Development forecasting experiments

Run 2026-10-24 via `notebooks/02-forecasting-methods.ipynb`. Compared six methods on 50 training-selected UK products, ten development origins and four horizons: 2,000 forecast targets per method. Origins overlap; no final-test forecast evaluation has run.

Lowest development MASE: **lag_boosting_7**, 0.6622. Seasonal naive: 1.0710. Prophet: 0.7296. The four-week moving average has MASE 0.6643; the best method improves on it by only 0.31%. These are exploratory point-forecast results, not a production promotion or measured inventory savings. See the notebook's paired origin/horizon plots and product failures, plus `forecasting-development-metrics.csv` and `forecasting-development.json` for exact metrics and lineage.

Seasonal fallback targets: 0; annual-enabled Prophet fits: 500 / 500. Missing 52-week lags use native tree handling. Forecasts are clipped nonnegative; customer identity, actual future prices and future-week sales are not inputs.

Passed executable leakage, fallback, arithmetic and comparison-population checks. Next: inspect instability and duplicate/extreme-order sensitivity, calibrate cumulative historical uncertainty, compare inventory scenarios, and freeze a candidate before final evaluation. Full promotion gates remain unassessed.
