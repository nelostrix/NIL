# query_ranker

Constructor in the shallow NIL catalog. See nil.help('ranking') for the workflow.

## Usage

```python
import nil
model = nil.model('query_ranker')
print(type(model).__name__)
```

## Parameters

See nil.help('ranking'); native parameters and validation remain explicit.

## Output

A model for the ranking workflow. Fit before predicting or saving.

## Requirements

Use the optional backend documented in nil.help('ranking'); the base install adds only NIL.

## Limitations

The supported scope, assumptions, persistence and validation limits are in nil.help('ranking'). This constructor does not imply every algorithm in that field is implemented.
