# forecasting

Direct univariate lag/horizon regression and chronological evaluation.

## Usage

```python
import nil
model = nil.forecasting.fit(list(range(60)), lags=6, horizon=3)
print(model.predict())
report = nil.forecasting.backtest(list(range(80)), initial_train=40, lags=6, horizon=3, gap=2, step=7)
print(report['mae'])
```

## Parameters

fit(series, lags=12, horizon=1, estimator=None, dates=None). Dates are ISO YYYY-MM-DD strings at a fixed positive day cadence. date_features(dates) returns ordinal/day-of-week/month cycles. backtest accepts initial_train, horizon, step, gap and max_fits.

## Output

A serializable lag_forecaster; predict(steps=None) returns up to its fitted direct horizon. Backtests return origin boundaries, predictions, actuals and MAE/RMSE.

## Requirements

nil-sdk[classical]. Date parsing uses the standard library.

## Limitations

Univariate fixed-horizon direct forecasts, not ARIMA/ETS/state-space, irregular sampling, probabilistic forecasting or hierarchical reconciliation. The default is Ridge; custom estimators must handle a matrix target for multiple horizons. No implicit frequency inference/resampling. Backtests clone/refit each prefix and exclude the explicit gap. Overlapping horizons count once per origin, not once per unique timestamp. Date predictors are known calendar information; no target-derived future features. Save with nil.save and reload with trusted=True.
