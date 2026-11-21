# kaplan_meier

Constructor in the shallow NIL catalog. See nil.help('survival') for the workflow.

## Usage

```python
import nil
model = nil.model('kaplan_meier')
print(type(model).__name__)
```

## Parameters

See nil.help('survival'); native parameters and validation remain explicit.

## Output

A model for the survival workflow. Fit before predicting or saving.

## Requirements

Use the optional backend documented in nil.help('survival'); the base install adds only NIL.

## Limitations

The supported scope, assumptions, persistence and validation limits are in nil.help('survival'). This constructor does not imply every algorithm in that field is implemented.
