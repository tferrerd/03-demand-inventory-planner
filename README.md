# 03 — Demand and Inventory Planner

- **Application:** Demand forecasting
- **Portfolio project:** 03 of 15
- **Status:** Source-derived CSV data is committed. ML models, the application and MLOps remain to be implemented.

Part of [Tomas Ferrer's DS/ML portfolio](https://tomasferrer.com/).

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
- [ ] Data audit, project charter, success criteria and evaluation protocol complete.
- [ ] Baseline and trained ML component evaluated; artifacts and failure analysis saved.
- [ ] Working application deployed with the exact evaluated model version.
- [ ] Tracking, registry, CI/CD, monitoring, retraining and rollback demonstrated.
- [ ] Stakeholder memo, ownership/operating record, case study and presentation complete.

## Current implementation

Published dataset observations, CSV exports, supporting assets where applicable, and provenance/checksum documentation are present. Models, application functionality and MLOps have not been implemented or evaluated yet.
