# item_cosine

Constructor in the shallow NIL catalog. See nil.help('recommendation') for the workflow.

## Usage

```python
import nil
model = nil.model('item_cosine', neighbors=20)
print(type(model).__name__)
```

## Parameters

See nil.help('recommendation'); native parameters and validation remain explicit.

## Output

A model for the recommendation workflow. Fit before predicting or saving.

## Requirements

Use the optional backend documented in nil.help('recommendation'); the base install adds only NIL.

## Limitations

The supported scope, assumptions, persistence and validation limits are in nil.help('recommendation'). This constructor does not imply every algorithm in that field is implemented.
