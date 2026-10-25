# Ownership and operating record

## 2026-10-25 — Cumulative uncertainty and inventory scenarios

Implemented and executed `notebooks/03-inventory-decisions.ipynb` with Codex assistance after documenting the protocol amendment. Reused the notebook 02 feature/fit recipe through `scripts/forecasting.py`; rerunning notebook 02 reproduced all 12,000 previous predictions. Added cumulative calibration and scenario helpers in `scripts/inventory.py`, with synthetic regression checks for future-outcome exclusion, pooling, quantile arithmetic, order rounding, cost arithmetic, invalid inputs and block resampling.

Replayed 49 origins per data variant and two methods over the fixed training-selected universe. Compared 500 development product-origins per method/variant. Saved 58,224 weekly predictions, 3,000 calibrated development totals, 78,000 exact calibration links and 36,000 scenario orders. Every calibration outcome ends before its order date; no final-test predictions or metrics were computed.

Original-sales 80% coverage: mean 80.2%, boosting 79.8%. Nominal 95% coverage: 87.8% and 88.4%, so tail calibration is inadequate. For Cu=4, Co=1, zero inventory, boosted versus mean quantile ordering has mean paired cost difference -136.79 arbitrary units per decision (11.1% lower), with a four-origin-block bootstrap interval [-193.22, +2.17]. Quantile-versus-point mean-cost reductions are exploratory and their primary paired intervals also cross zero. No dependable savings, production model or completed promotion is claimed.

Sensitivities consistently refit/recalibrate after exact-repeat removal (682 selected-product lines, 5,136 units removed before final evaluation) and under a fixed counterfactual 1,000-unit line cap (98 lines, 127,860 units capped). The cap is a diagnostic alternate target, not a source correction. Source CSVs and final-evaluation boundaries remain unchanged. Scenarios assume immediate order availability for independent four-week windows; no actual supplier timing or inventory logs are available.

Next: review calibration failures and robustness, amend development methodology if justified, freeze the recipe before final evaluation, then complete model artifacts/tracking, application/deployment, monitoring and rollback. All current outputs remain offline development evidence.

## 2026-10-24 — Initial forecasting experiments

Expanded notebook 01 with original source-row previews, a field dictionary, examples explaining filtering decisions, and a worked trace from one product's invoice lines to weekly forecasting targets. The walkthrough reuses the checked audit loader and cleaning rules and asserts that recomputed quantities match every saved week for the example product. This adds visibility into existing transformations; no source observations, selection rules or forecast results are changed.

Added and executed `notebooks/02-forecasting-methods.ipynb` with Codex assistance. Documented the protocol amendment before comparing models. Fitted 500 product/origin Prophet models and 20 pooled histogram boosting models, alongside last-week, four-week-mean and annual seasonal-naive baselines. All methods share 50 products, ten development origins and four weekly horizons, yielding 12,000 saved prediction rows across six methods. No final-test predictions or final-test metrics were computed.

Development MASE: seven-leaf boosting 0.6622; fifteen-leaf boosting 0.6635; four-week mean 0.6643; Prophet 0.7296; last week 0.7964; seasonal naive 1.0710. The best tree model's approximately 0.3% advantage over the moving average is exploratory; the mean has slightly lower unit MAE. No model was promoted. There is no measured inventory benefit.

Executed checks cover origin-feature and fitting-row future invariance, multi-step lag anchoring, annual lookups/fallback, metric arithmetic, identical forecast populations, pre-origin fitting data and nonnegative finite outputs. Saved paired predictions, fit diagnostics, product errors, summary metrics and code/data/environment lineage. Notebook outputs include plots and product failure cases. Existing audit regression checks pass.

The local Python 3.9 environment uses pinned Prophet 1.1.7 and compatible CmdStanPy 1.2.5; the newer backend initially failed initialization and was replaced with the compatible version. Source observations and the final-evaluation protocol were not changed. Interval calibration, duplicate/extreme-order sensitivity, inventory simulation, final evaluation, fitted release artifacts, tracking/registration, deployment, monitoring and rollback remain outstanding.

## 2026-10-21 — Initial milestone

Implemented with Codex assistance: checksum/schema validation for all committed CSV exports; documented later-sheet ownership of overlapping dates; UK positive-sales construction; training-only product selection; weekly panel and four-week rolling origins; project charter and evaluation protocol; executable exploration notebook and regression checks. Tomas Ferrer's portfolio project; no external stakeholder approval or business adoption is claimed.

Validation commands and exact descriptive findings are recorded in the README and generated audit report. The audit records source, configuration, code and generated-file hashes plus runtime versions. Raw source files are unchanged. Derived interim/processed CSVs are reproducible and excluded from Git.

No fitted demand model, forecast evaluation, MLflow run, registered release, live application, production traffic, refresh exercise or rollback has been demonstrated. No real inventory cost or service improvement has been measured. Notebook plots are descriptive, not predictive results. The original workbook acquisition/re-extraction script remains outstanding; current reproducibility starts with the committed exports.

Next milestone: implement and evaluate seasonal naive, moving average and the trained lag-feature candidate under the frozen protocol; save paired forecasts, cumulative interval calibration and simulated policy comparisons. Then build the application and operational lifecycle around the evaluated artifact, followed by deployment evidence, stakeholder memo, presentation and case study.
