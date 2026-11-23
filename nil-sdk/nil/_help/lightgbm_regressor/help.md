# lightgbm_regressor

Constructor in the shallow NIL catalog. See nil.help('boosting') for the workflow.

## Usage

```python
import nil
model = nil.model('lightgbm_regressor', n_estimators=20)
print(type(model).__name__)
```

## Parameters

See nil.help('boosting'); native parameters and validation remain explicit.

## Output

A model for the boosting workflow. Fit before predicting or saving.

## Requirements

Use the optional backend documented in nil.help('boosting'); the base install adds only NIL.

## Limitations

The supported scope, assumptions, persistence and validation limits are in nil.help('boosting'). This constructor does not imply every algorithm in that field is implemented.
