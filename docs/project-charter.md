# Project charter

Prepared 2026-10-21, before fitting a forecaster.

The Demand and Inventory Planner helps a small online retailer's inventory operator compare weekly replenishment decisions for established UK products. The first release forecasts four weeks of recorded positive unit sales and explains how uncertainty changes a hypothetical order quantity. The source is a historical benchmark, not a client engagement.

The operator reviews a product at the beginning of Monday, using transactions recorded through the preceding Sunday. The initial population is up to 50 established UK product series, selected by training-period units after minimum-history checks. Other countries, new products and customer-level demand are outside the first milestone. Stock codes follow a documented heuristic rather than a supplied catalog.

The current forecasting comparator is annual seasonal naive on weekly data: use the target week's sales 52 weeks earlier, falling back to the last observed week when that history is unavailable. Compare a fitted lag-feature forecaster with this baseline on identical origins and products. A fitted forecast and its calibrated distribution will change the policy's order-up-to quantity; an unsuccessful candidate stays an experimental result and cannot replace the baseline.

## Decision and costs

Primary business KPI: mean **simulated cost per product-origin**, using the same inventory position, lead time and cost inputs for both policies. Report total cost, shortage units, surplus units, units sold, product/origin counts, dates and comparator. Monetary units are scenario units; transaction sales prices do not identify purchase, holding or shortage costs.

The initial decision is a single-period, four-week protection-window order under a stated lost-sales approximation. For nonnegative scenario shortage penalty `Cu` and surplus penalty `Co` with positive sum, choose the cumulative-sales quantile at `Cu / (Cu + Co)`, then order `max(0, ceil(quantile - inventory_position))`. Inventory position is on-hand plus on-order minus committed units, supplied by the user. Initially require nonnegative inventory position and no intra-window receipts; extensions need an explicit time-stepped simulator. Excess ordering incurs surplus cost; insufficient ordering incurs shortage cost. No actual inventory optimization or company savings can be inferred without inventory and availability logs.

Guardrails: distinguish sales from unconstrained demand; exclude returns from the positive-sales target; report zeros and extreme orders; no future-actual lag features; no final-test model tuning; no unsupported net-revenue claims. Negative prices and non-product adjustments must not enter product sales. Missing customer IDs are retained because customer identity is unnecessary for this target. A historical zero does not establish an available product with zero demand.

## Measurable success criteria

1. One audit command verifies every committed source file and regenerates provenance-linked weekly series, training-only product selection and temporal origins.
2. Evaluate seasonal naive and a trained lag-feature regressor on the same development and final four-week origins; publish scaled error, quantile loss, interval coverage and simulated costs, including negative results.
3. Apply the predefined development and final release gates in the evaluation plan. Calibrate cumulative intervals on completed historical forecast windows, never future residuals.
4. Make the evaluated model affect an application output. Tracking, registration, deployment, monitoring, refresh and rollback must be demonstrated before declaring the full project complete.
5. Deliver a decision memo, presentation and case study that distinguish measured forecast performance, hypothetical policy costs and evidence requiring a real rollout.

Delivery sequence: data audit and protocol first; baseline and trained comparison second; application and MLOps third; operating evidence and communication artifacts last. The first milestone makes no forecast-accuracy or business-improvement claim.
