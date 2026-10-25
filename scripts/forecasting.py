"""Shared historical feature construction and lag-model recipe for notebooks 02 and 03."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

FEATURES = ['StockCode', 'horizon', 'calendar_sin', 'calendar_cos'] + [f'lag_{k}' for k in [1, 2, 4, 8, 13, 52]] + [f'{name}_{k}' for k in [4, 13] for name in ['mean', 'zero_fraction']]


def make_design(data, horizon=4):
    rows = []
    for sku, group in data.groupby('StockCode', sort=True):
        series = group.set_index('week').units.sort_index()
        assert series.index.equals(pd.date_range(series.index.min(), series.index.max(), freq='W-MON'))
        values = series.to_numpy(dtype=float)
        for i in range(13, len(series)):
            origin = series.index[i]
            past = values[:i]
            features = {'StockCode': sku, 'origin': origin, 'history_through': origin - pd.Timedelta(days=1)}
            for lag in [1, 2, 4, 8, 13, 52]:
                features[f'lag_{lag}'] = past[-lag] if len(past) >= lag else np.nan
            for window in [4, 13]:
                features[f'mean_{window}'] = past[-window:].mean()
                features[f'zero_fraction_{window}'] = (past[-window:] == 0).mean()
            for h in range(1, horizon + 1):
                target = origin + pd.Timedelta(weeks=h - 1)
                angle = 2 * np.pi * (target.dayofyear - 1) / 365.25
                rows.append({**features, 'horizon': h, 'target_week': target,
                             'calendar_sin': np.sin(angle), 'calendar_cos': np.cos(angle),
                             'actual': values[i + h - 1] if i + h - 1 < len(values) else np.nan})
    return pd.DataFrame(rows)


def baseline_predictions(history, target_dates):
    assert history.index.max() < min(target_dates)
    last = float(history.iloc[-1])
    prior_year = pd.DatetimeIndex(target_dates) - pd.Timedelta(weeks=52)
    seasonal = history.reindex(prior_year)
    return {'last_week': np.repeat(last, len(target_dates)),
            'mean_4_weeks': np.repeat(float(history.tail(4).mean()), len(target_dates)),
            'seasonal_naive': seasonal.fillna(last).to_numpy()}, seasonal.isna().to_numpy()


def fit_lag_model(fitting, target_rows, leaves=7, seed=20261021):
    """Fit on completed targets; retain native handling of missing long lags."""
    if fitting.empty or target_rows.empty:
        raise ValueError("Nonempty fitting and forecast rows required")
    if fitting.target_week.max() >= target_rows.origin.min():
        raise ValueError("Fitting targets must precede forecast origin")
    categories = sorted(fitting.StockCode.unique())
    fit_x, predict_x = fitting[FEATURES].copy(), target_rows[FEATURES].copy()
    for x in [fit_x, predict_x]:
        x['StockCode'] = pd.Categorical(x.StockCode, categories=categories)
        if x.StockCode.isna().any():
            raise ValueError("Forecast SKU absent from fitting history")
    model = HistGradientBoostingRegressor(loss='squared_error', max_leaf_nodes=leaves,
                learning_rate=.05, max_iter=150, min_samples_leaf=20, l2_regularization=1.,
                categorical_features='from_dtype', early_stopping=False, random_state=seed)
    with threadpool_limits(limits=2):
        model.fit(fit_x, fitting.actual)
        estimate = np.maximum(model.predict(predict_x), 0)
    return model, estimate
