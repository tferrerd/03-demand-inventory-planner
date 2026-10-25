"""Historical cumulative calibration and explicit single-window inventory scenarios."""

import numpy as np
import pandas as pd

from forecasting import make_design, fit_lag_model

QUANTILES = {"q025": .025, "q10": .1, "q50": .5, "q80": .8, "q90": .9, "q975": .975}
METHODS = ["mean_4_weeks", "lag_boosting_7"]


def forecast_history(panel, origins, variant):
    """Replay both fixed recipes, requiring 26 preceding weeks per product."""
    design = make_design(panel)
    first_weeks = panel.groupby("StockCode").week.min()
    output = []
    for origin in origins:
        targets = design.loc[design.origin.eq(origin)].copy()
        history_weeks = (targets.origin - targets.StockCode.map(first_weeks)).dt.days // 7
        targets = targets.loc[history_weeks.ge(26)].sort_values(["StockCode", "horizon"])
        if targets.empty:
            continue
        if targets.actual.isna().any() or (targets.groupby("StockCode").size() != 4).any():
            raise ValueError("Every historical forecast requires four observed target weeks")
        fitting = design.loc[design.target_week.lt(origin) & design.actual.notna()]
        _, tree = fit_lag_model(fitting, targets, leaves=7)
        base = targets[["StockCode", "origin", "horizon", "target_week", "actual", "mean_4", "mean_13"]].copy()
        for method, predictions in [("mean_4_weeks", targets.mean_4.to_numpy()), ("lag_boosting_7", tree)]:
            rows = base.assign(variant=variant, method=method, prediction=predictions)
            rows["fit_max_target_week"] = fitting.target_week.max() if method == "lag_boosting_7" else origin - pd.Timedelta(weeks=1)
            output.append(rows)
    if not output:
        raise ValueError("No forecasts available")
    return pd.concat(output, ignore_index=True)


def cumulative_forecasts(weekly):
    keys = ["variant", "method", "StockCode", "origin"]
    if weekly.duplicated(keys + ["horizon"]).any() or not weekly.groupby(keys).size().eq(4).all():
        raise ValueError("Expected four unique weekly horizons per cumulative forecast")
    result = weekly.groupby(keys, as_index=False).agg(
        actual=("actual", "sum"), prediction=("prediction", "sum"),
        last_week=("target_week", "max"), mean_4=("mean_4", "first"), mean_13=("mean_13", "first"))
    result["target_end"] = result.last_week + pd.Timedelta(days=6)
    result["residual"] = result.actual - result.prediction
    result["residual_scale"] = 4 * result.mean_13.clip(lower=1)
    result["forecast_id"] = result.variant + "/" + result.method + "/" + result.StockCode + "/" + result.origin.dt.strftime("%Y-%m-%d")
    return result


def calibrate_one(row, historical):
    """Only completed windows can determine this forecast's empirical quantiles."""
    past = historical.loc[
        historical.variant.eq(row.variant) & historical.method.eq(row.method)
        & historical.target_end.lt(row.origin)
    ].sort_values(["origin", "StockCode"])
    own = past.loc[past.StockCode.eq(row.StockCode)].tail(26)
    if len(own) >= 12:
        members = own
        errors = members.residual.to_numpy()
        source = "product"
    else:
        latest_origins = past.origin.drop_duplicates().tail(26)
        members = past.loc[past.origin.isin(latest_origins)]
        if len(members) < 12 or members.origin.nunique() < 4:
            raise ValueError("Insufficient completed history for pooled calibration")
        errors = (members.residual / members.residual_scale * row.residual_scale).to_numpy()
        source = "pooled_normalized"
    if not np.isfinite(errors).all() or not np.isfinite(row.prediction) or row.prediction < 0:
        raise ValueError("Calibration inputs must be finite; prediction must be nonnegative")
    values = np.maximum.accumulate(np.maximum(0, row.prediction + np.quantile(errors, list(QUANTILES.values()), method="linear")))
    estimate = dict(zip(QUANTILES, values))
    estimate.update({"calibration_source": source, "calibration_rows": len(members),
                     "calibration_origins": members.origin.nunique(),
                     "calibration_latest_target_end": members.target_end.max()})
    membership = pd.DataFrame({"forecast_id": row.forecast_id,
                               "calibration_forecast_id": members.forecast_id,
                               "calibration_target_end": members.target_end,
                               "calibration_origin": members.origin,
                               "used_error_current_units": errors})
    return estimate, membership


def inventory_outcome(actual, target, inventory, cu, co):
    """Independent protection window; order assumed available before its sales."""
    actual, target, inventory = np.broadcast_arrays(np.asarray(actual, dtype=float), np.asarray(target, dtype=float), np.asarray(inventory, dtype=float))
    if (not np.isfinite([cu, co]).all() or cu < 0 or co < 0 or cu + co <= 0
            or not all(np.isfinite(x).all() and (x >= 0).all() for x in [actual, target, inventory])):
        raise ValueError("Finite nonnegative units/costs and positive total penalty required")
    order = np.maximum(0, np.ceil(target - inventory))
    available = inventory + order
    shortage = np.maximum(actual - available, 0)
    surplus = np.maximum(available - actual, 0)
    return {"order": order, "available": available, "shortage": shortage,
            "surplus": surplus, "cost": cu * shortage + co * surplus}


def simulate_policies(calibrated):
    output = []
    for cu, quantile in [(1, "q50"), (4, "q80"), (9, "q90")]:
        for inventory_name in ["zero", "two_weeks_mean"]:
            inventory = np.zeros(len(calibrated)) if inventory_name == "zero" else np.ceil(2 * calibrated.mean_4.to_numpy())
            for policy in ["point", "quantile"]:
                target = calibrated.prediction.to_numpy() if policy == "point" else calibrated[quantile].to_numpy()
                outcomes = inventory_outcome(calibrated.actual, target, inventory, cu, 1)
                rows = calibrated[["variant", "method", "StockCode", "origin", "actual", "forecast_id"]].copy()
                rows = rows.assign(policy=policy, Cu=cu, Co=1, alpha=cu / (cu + 1),
                                   inventory_scenario=inventory_name, initial_inventory=inventory,
                                   target=target, **outcomes)
                output.append(rows)
    return pd.concat(output, ignore_index=True)


def summarize_policies(policies):
    keys = ["variant", "method", "policy", "Cu", "Co", "inventory_scenario"]
    rows = []
    for key, group in policies.groupby(keys):
        denominator = float(group.actual.sum())
        rows.append(dict(zip(keys, key), product_origins=len(group),
                         total_cost=float(group.cost.sum()), mean_cost=float(group.cost.mean()),
                         shortage_units=float(group.shortage.sum()), surplus_units=float(group.surplus.sum()),
                         mean_order_units=float(group.order.mean()),
                         shortage_event_fraction=float(group.shortage.gt(0).mean()),
                         actual_units_denominator=denominator,
                         simulated_unit_fulfillment=1 - float(group.shortage.sum()) / denominator if denominator else np.nan))
    return pd.DataFrame(rows)


def summarize_intervals(calibrated):
    rows = []
    for (variant, method), group in calibrated.groupby(["variant", "method"]):
        row = {"variant": variant, "method": method, "product_origins": len(group),
               "cumulative_MAE": float((group.actual - group.prediction).abs().mean()),
               "coverage80": float(group.actual.between(group.q10, group.q90).mean()),
               "coverage95": float(group.actual.between(group.q025, group.q975).mean()),
               "width80": float((group.q90 - group.q10).mean()),
               "width95": float((group.q975 - group.q025).mean()),
               "pooled_fallback_rows": int(group.calibration_source.eq("pooled_normalized").sum()),
               "minimum_calibration_rows": int(group.calibration_rows.min()),
               "maximum_calibration_rows": int(group.calibration_rows.max())}
        for name, alpha in [("q10", .1), ("q50", .5), ("q90", .9)]:
            error = group.actual - group[name]
            row["pinball_" + name] = float(np.maximum(alpha * error, (alpha - 1) * error).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def block_mean_interval(values, replicates=500, block_length=4, seed=20261021):
    values = np.asarray(values, dtype=float)
    if len(values) < block_length or not np.isfinite(values).all():
        raise ValueError("Need finite ordered values and at least one full block")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values) - block_length + 1,
                          size=(replicates, int(np.ceil(len(values) / block_length))))
    indices = (starts[:, :, None] + np.arange(block_length)).reshape(replicates, -1)[:, :len(values)]
    return np.quantile(values[indices].mean(axis=1), [.025, .975])


def paired_cost_comparisons(policies):
    rows, per_origin = [], []
    pairs = [("mean_4_weeks/quantile", "mean_4_weeks/point"),
             ("lag_boosting_7/quantile", "lag_boosting_7/point"),
             ("lag_boosting_7/quantile", "mean_4_weeks/quantile"),
             ("lag_boosting_7/point", "mean_4_weeks/point")]
    keys = ["variant", "Cu", "Co", "inventory_scenario"]
    for key, group in policies.groupby(keys):
        group = group.assign(choice=group.method + "/" + group.policy)
        table = group.pivot(index=["StockCode", "origin"], columns="choice", values="cost")
        if table.isna().any().any():
            raise ValueError("Paired comparisons require identical forecast populations")
        for candidate, comparator in pairs:
            differences = table[candidate] - table[comparator]
            ordered = differences.groupby("origin").mean().sort_index()
            low, high = block_mean_interval(ordered.to_numpy())
            baseline = float(table[comparator].mean())
            row = dict(zip(keys, key), candidate=candidate, comparator=comparator,
                       mean_cost_difference=float(differences.mean()), bootstrap_low=float(low), bootstrap_high=float(high),
                       comparator_mean_cost=baseline,
                       cost_reduction_pct=-100 * float(differences.mean()) / baseline if baseline else np.nan,
                       origins=len(ordered), product_origins=len(differences))
            rows.append(row)
            for origin, value in ordered.items():
                per_origin.append(dict(zip(keys, key), candidate=candidate, comparator=comparator,
                                       origin=origin, mean_cost_difference=float(value)))
    return pd.DataFrame(rows), pd.DataFrame(per_origin)
