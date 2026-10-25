# 03 — Demand and Inventory Planner

- **Application:** Demand forecasting
- **Portfolio project:** 03 of 15
- **Status:** Data audit, development forecasting, cumulative uncertainty calibration and simulated inventory-policy comparisons are implemented. Tail calibration needs improvement; final evaluation, the application and MLOps remain outstanding.

Part of [Tomas Ferrer's DS/ML portfolio](https://tomasferrer.com/).

## Start here

The first milestone (2026-10-21) establishes a recorded-sales forecasting benchmark before model fitting:

- [Exploration notebook](notebooks/01-exploration.ipynb): original transaction rows, field meanings, exceptional-record examples and a worked invoice-line-to-weekly-sales calculation; then source overlap, missing fields, product scope, intermittency and temporal evaluation windows.
- [Forecasting methods notebook](notebooks/02-forecasting-methods.ipynb): paired development backtests of last-week, moving-average, seasonal-naive, Prophet and pooled lag-feature forecasts. Select `.venv/bin/python` as the kernel after installing requirements. Includes leakage checks, horizon/product diagnostics and saved predictions; final evaluation and calibrated inventory policies remain separate.
- [Inventory decisions notebook](notebooks/03-inventory-decisions.ipynb): completed-window cumulative calibration, forecast-range coverage, point-versus-quantile ordering, six cost/inventory scenarios, paired origin-block intervals and duplicate/bulk-order sensitivities. Run after notebook 02; no final-test forecasts or promotion.
- [Project charter](docs/project-charter.md): operator, ordering decision, simulated cost KPI and success criteria.
- [Evaluation plan](docs/evaluation-plan.md): seasonal-naive baseline, lag-feature candidate, historical interval calibration, rolling-origin evaluation and promotion gates.
- [Data audit](docs/reports/01-data-audit.md) and [machine-readable evidence](docs/reports/data-audit.json).
- [Configuration](config.json) and [operating record](docs/operating-record.md).

### Reproduce the first milestone

Python 3.9 or later; the recorded run used Python 3.9.6. From this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/audit.py
python -m unittest discover -s tests -v
```

Open `notebooks/01-exploration.ipynb` and run all cells after the audit. To execute it without the notebook UI:

```bash
python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
path = Path("notebooks/01-exploration.ipynb")
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=180, kernel_name="python3").execute()
nbformat.write(notebook, path)
PY
```

The audit checks SHA-256, sizes, row counts, schema and required values for all 12 committed CSVs. It regenerates `data/interim/uk-sales.csv`, `data/processed/weekly-sales.csv`, `selected-products.csv` and `forecast-origins.csv`; these derived CSVs are excluded from Git. The interim sales file preserves source-file and zero-based CSV-data-row references. Reports record runtime versions and source/config/code/output hashes. Reproduction currently begins with committed exports; original-workbook acquisition and re-extraction remain outstanding.

Actual findings: **1,067,371 raw rows**, with **22,523 earlier-sheet rows removed** from overlapping December 2010 dates. Retain 11,812 exact repeated rows after sheet reconciliation pending sensitivity analysis. The full-week UK positive-sales scope contains 905,162 lines and 8,850,606 units. Training-only eligibility/ranking selects **50 products**, representing 21.93% of eligible training units. Their **5,097 product-weeks** contain 6.83% zeros; 3,797 rows are training and 650 each are development and test. There are ten four-week origins in each evaluation partition.

These are recorded sales, not unconstrained demand. Returns and non-product adjustments are excluded from the positive-sales target; missing customer IDs are retained. Zero-sales weeks do not identify stockouts or available inventory. Initial development forecast accuracy is recorded below; no final-test performance, real savings or production operation has been measured. Inventory costs and lead times will be explicit scenario assumptions.

### Initial forecasting experiments — 2026-10-24

Notebook 02 compares six methods on the same 50 products, ten development origins and four weekly horizons (2,000 targets per method). Mean absolute scaled error (MASE; lower is better):

| Method | Development MASE |
|---|---:|
| Lag boosting, 7 leaves | 0.6622 |
| Lag boosting, 15 leaves | 0.6635 |
| Four-week moving average | 0.6643 |
| Prophet | 0.7296 |
| Last week | 0.7964 |
| Annual seasonal naive | 1.0710 |

The best boosted model improves MASE only about **0.3% over the moving average**, which has slightly lower unit MAE. Prophet does not lead this comparison. Overlapping windows and development selection prevent a confirmatory superiority claim. Cumulative error also changes the model ranking. See the [experiment report](docs/reports/02-forecasting-methods.md), [metrics](docs/reports/forecasting-development-metrics.csv) and [run lineage](docs/reports/forecasting-development.json). Leakage checks and paired-population assertions pass. Interval calibration, inventory costs, robustness, final evaluation and promotion remain outstanding.

Use the project virtual environment as the notebook kernel. Prophet 1.1.7 and its compatible CmdStanPy 1.2.5 backend are pinned in requirements for the recorded Python 3.9 run. Notebook 02 saves development predictions, per-product errors and fit diagnostics under `data/processed/`; compact method/horizon/origin summaries are saved under `docs/reports/`.

### Uncertainty and hypothetical inventory decisions — 2026-10-25

Notebook 03 replays both the moving average and seven-leaf model at 49 historical origins for the fixed product universe, calibrating each development forecast only on completed prior four-week windows. It compares 500 product-origins per method across three sales definitions and saves 36,000 hypothetical orders plus 78,000 exact calibration-membership links.

On original sales, nominal 80% cumulative ranges cover 80.2% of outcomes for the moving average and 79.8% for boosting. Nominal 95% ranges cover only 87.8% and 88.4%; **tail calibration needs work**. Under the Cu=4, Co=1, zero-inventory scenario, boosted quantile ordering averages 1,093.99 arbitrary cost units per decision versus 1,230.78 for mean quantile ordering (11.1% lower). The paired origin-block interval for their cost difference is [-193.22, +2.17], including no improvement. This is post-selection development simulation with ten overlapping origins, not dependable savings or production evidence.

Both methods are refitted/recalibrated after exact-repeat removal and under a labeled counterfactual 1,000-unit invoice-line cap. The order is assumed available before an independent four-week window; supplier delays and ongoing inventory are not simulated. See the [inventory report](docs/reports/03-inventory-decisions.md), [coverage metrics](docs/reports/inventory-interval-metrics.csv), [policy metrics](docs/reports/inventory-policy-metrics.csv) and [paired comparisons](docs/reports/inventory-paired-comparisons.csv). Full replay artifacts are regenerable in `data/processed/`. Calibration/ordering regression checks run with the existing unittest command.

## Committed CSV dataset

**Dataset:** UCI Online Retail II

**Coverage:** Complete workbook observations from all sheets, preserving sheet/source order and split into CSV files below GitHub's file-size limit.

| CSV file | Rows |
|---|---:|
| [retail_sheet1_001.csv](data/raw/retail_sheet1_001.csv) | 100,000 |
| [retail_sheet1_002.csv](data/raw/retail_sheet1_002.csv) | 100,000 |
| [retail_sheet1_003.csv](data/raw/retail_sheet1_003.csv) | 100,000 |
| [retail_sheet1_004.csv](data/raw/retail_sheet1_004.csv) | 100,000 |
| [retail_sheet1_005.csv](data/raw/retail_sheet1_005.csv) | 100,000 |
| [retail_sheet1_006.csv](data/raw/retail_sheet1_006.csv) | 25,461 |
| [retail_sheet2_007.csv](data/raw/retail_sheet2_007.csv) | 100,000 |
| [retail_sheet2_008.csv](data/raw/retail_sheet2_008.csv) | 100,000 |
| [retail_sheet2_009.csv](data/raw/retail_sheet2_009.csv) | 100,000 |
| [retail_sheet2_010.csv](data/raw/retail_sheet2_010.csv) | 100,000 |
| [retail_sheet2_011.csv](data/raw/retail_sheet2_011.csv) | 100,000 |
| [retail_sheet2_012.csv](data/raw/retail_sheet2_012.csv) | 41,910 |

These are actual published observations or documented format conversions. See [source, license and limitations](data/DATASET.md) and [CSV schemas/checksums](data/csv-manifest.json).

## Project prompt

Build an end-to-end demand forecasting application that turns a published online dataset into a useful business decision. Implement the analysis, a trained machine learning component, a deployed application, its MLOps lifecycle, and a presentation. The learned model must influence the application's output. Begin with the dataset and adapt the scope to the evidence it actually contains.

### Business question and application

**Decision:** How much inventory should an operator order, given uncertain demand and asymmetric shortage costs?

**Application to build:** Select a product and country/cohort, inspect forecast intervals, change lead time and inventory costs, and compare suggested replenishment policies.

Write a short project charter defining the user, current baseline, primary business KPI, guardrails, prediction time, and measurable success criteria before selecting a model. Explain which decision the model changes and the costs of incorrect decisions.

### Published dataset and raw data

The committed source is **UCI Online Retail II**. Complete workbook observations from all sheets, preserving sheet/source order and split into CSV files below GitHub's file-size limit. See [dataset documentation](data/DATASET.md) for original links, attribution, usage terms and limitations.

Acquire a pinned source release into `data/raw/` and preserve the original bytes. Add a repeatable acquisition script and a manifest containing source URL, release/version, retrieval time, citation, terms, file sizes and SHA-256 checksums. Keep cleaned data in `data/interim/` and model-ready features/split assignments in `data/processed/`. Redistributable CSV exports and the documented supporting assets are committed; large source archives remain excluded from Git.

Audit labels, decision-time feature availability, duplicates, missingness, class balance, and temporal/group structure. If the candidate cannot support the task or intended use, find another suitable published dataset or narrow the question. Generated fixtures and economic scenarios are supplementary, clearly labeled artifacts.

### Machine learning component

**Required fitted component:** Trained lag-feature demand forecaster.

Seasonal naive baseline versus lag-feature regression/boosted trees; add prediction intervals and evaluate their coverage.

Produce a reproducible training command and save the fitted preprocessing, model and decision policy together where appropriate. Compare the learned approach with a simple baseline. Pretrained models may support the solution, but a pretrained API call or prompt alone does not fulfill the trained-component requirement.

### Evaluation and business impact

Rolling-origin backtests, MASE or RMSSE, quantile loss, interval coverage, and simulated inventory cost. Fit preprocessing and interval calibration only on the permitted historical data.

Separate development/model selection from final evaluation. Fit transformations on training data only and compare candidate and incumbent on the same evaluation protocol. Save metrics, uncertainty/robustness checks, and representative failure cases with code/data/model identifiers.

Connect model metrics to the business KPI, with explicit denominators, comparator, population and period. Distinguish measured service performance, offline evaluation, simulated policy value and outcomes that require a real rollout. Never invent company impact or actual stakeholder buy-in.

### MLOps requirements

**Project-specific operational evidence:** Train against historical cutoffs; version forecast horizon and interval calibration; monitor error and interval coverage as actuals arrive.

Implement and demonstrate:

1. Data/schema validation and versioned acquisition, features, splits and configuration.
2. Reproducible training with MLflow experiment tracking and saved metrics/artifacts.
3. Model registration with code/data lineage, exact versions, candidate/incumbent comparison and promotion gates.
4. Production inference with validated inputs and meaningful tests, including consistency with the evaluated model.
5. Containerized deployment and GitHub Actions CI/CD, with a pinned model/image, health check, inference smoke test and release identifier.
6. Service telemetry and task-specific input/prediction/quality monitoring, alert criteria and a response runbook. Identify replayed quality reports separately when real outcome labels are unavailable.
7. A runnable data-refresh/retraining workflow that evaluates the replacement before promotion.
8. A demonstrated rollback to the previous model/application release and a successful post-rollback inference check.

Use the smallest architecture that completes this lifecycle. Select real-time or batch inference according to the decision; shared managed services are acceptable, while the repository remains independently reproducible.

### Communication and portfolio deliverables

- A readable README with setup, actual results, architecture, limitations, and links to the released demo/case study/slides.
- Purpose-specific notebooks and short executive summaries documenting important experiments and decisions.
- A stakeholder decision memo covering the recommendation, evidence, tradeoffs and proposed rollout.
- A personal ownership statement and dated operating record identifying what was implemented, actual service usage, model changes, and incident/rollback exercises.
- A working public demo with source-derived examples where permitted, plus screenshots or a short recorded fallback.
- An original 10–12 slide presentation: editable `presentation/slides.pptx`, reviewed `presentation/slides.pdf`, speaker notes and underlying figures. Include model evaluation, business implications, MLOps architecture and operational evidence.
- A case study prepared for `tomasferrer.com`, linking the repository, demo and presentation.

**Presentation story:** The value of uncertainty estimates becomes visible when selecting an inventory policy.

**Distinctive evidence to include:** Baseline comparisons across multiple forecast origins and an explicit distinction between recorded sales and unconstrained demand.

## Completion checklist

- [x] Published source acquired, checksum verified and provenance/profile recorded.
- [x] Data audit, project charter, success criteria and evaluation protocol complete.
- [ ] Baseline and trained ML component evaluated; artifacts and failure analysis saved.
- [ ] Working application deployed with the exact evaluated model version.
- [ ] Tracking, registry, CI/CD, monitoring, retraining and rollback demonstrated.
- [ ] Stakeholder memo, ownership/operating record, case study and presentation complete.

## Current implementation

Published observations and provenance/checksum documentation are present. The checked audit, sheet-overlap reconciliation, training-only product selection, complete-week panel, rolling origins, exploration notebook and pre-model protocol are implemented. Notebook 02 fits Prophet and pooled lag-feature candidates and compares development point forecasts with three simple baselines. Notebook 03 implements completed-window cumulative calibration, inventory-policy scenarios and duplicate/bulk-order sensitivities. Tail calibration improvements, full promotion gates, final evaluation, original-workbook acquisition, application functionality and MLOps remain outstanding.
