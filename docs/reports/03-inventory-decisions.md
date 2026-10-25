# Inventory decisions under uncertainty

Executed 2026-10-25 via `notebooks/03-inventory-decisions.ipynb`. Two methods, 50 fixed UK products, ten development origins and three sales definitions. The seven-leaf model was selected on this same development period; all results remain exploratory. No final-test forecast evaluation or production promotion has run.

Original-sales 80% cumulative coverage: moving average 80.2%; boosted model 79.8%. The corresponding 95% coverage rates are 87.8% and 88.4%. Both nominal 95% ranges substantially under-cover; tail calibration needs further work. See interval metrics for widths, quantile losses, denominators and fallback counts. Every residual window ended before its new order date; exact membership is saved.

For the primary Cu=4, Co=1, zero-inventory scenario, boosted-quantile minus moving-average-quantile mean cost is -136.79 arbitrary cost units per product-origin. The paired four-origin-block bootstrap interval is [-193.22, 2.17], from 500 replicates over just ten overlapping origins. Negative favors boosting. This is simulated policy value against recorded sales, not realized savings or strong evidence of a dependable advantage.

All six cost/inventory scenarios and point-versus-quantile comparisons are saved. Both methods were refitted/recalibrated after exact-repeat removal and under a labeled counterfactual 1,000-unit invoice-line cap. Those variants change the sales target; compare rankings within each, not absolute profits between variants. Large uncapped orders and original interval failure cases appear in the notebook.

Next: review coverage and robustness failures, decide whether to amend the recipe, freeze it before final evaluation, and complete promotion gates. The scenario assumes orders available at the start of an independent four-week protection window; it does not model a supplier delay or ongoing inventory. No actual inventory/cost records, deployment, registry, monitoring or rollback evidence are supplied by this notebook.
