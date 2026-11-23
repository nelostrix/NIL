# decision_tree_regressor

Build `sklearn.tree.DecisionTreeRegressor` for continuous numeric targets.

## Usage

```python
import nil

model = nil.model("decision_tree_regressor", max_depth=3, random_state=7)
model.fit([[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 7.0])
print(model.predict([[4.0]]))
```

## Parameters

`max_depth`: maximum tree depth. `random_state`: reproducibility seed. Other arguments follow the native constructor.

## Output

A native `DecisionTreeRegressor` instance. Prediction/transform methods return native backend arrays.

## Requirements

Install `nil-sdk[classical]` or the source project with `pip install ".[classical]"`. Only this model's backend is loaded when constructed.

## Limitations

The tiny example demonstrates API use, not model quality. Prepare numeric features and validate on held-out data. NIL passes parameters directly to the backend and adds no per-prediction wrapper. Use native fit/predict or nil.fit/nil.evaluate. nil.train remains neural-specific.
