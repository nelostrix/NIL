# lag_forecaster

Constructor in the shallow NIL catalog. See nil.help('forecasting') for the workflow.

## Usage

```python
import nil
model = nil.model('lag_forecaster', lags=6, horizon=3)
print(type(model).__name__)
```

## Parameters

See nil.help('forecasting'); native parameters and validation remain explicit.

## Output

A model for the forecasting workflow. Fit before predicting or saving.

## Requirements

Use the optional backend documented in nil.help('forecasting'); the base install adds only NIL.

## Limitations

The supported scope, assumptions, persistence and validation limits are in nil.help('forecasting'). This constructor does not imply every algorithm in that field is implemented.
