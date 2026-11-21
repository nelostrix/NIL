# gradient_boosting_regressor

Build `sklearn.ensemble.HistGradientBoostingRegressor` for continuous numeric targets.

## Usage

```python
import nil

model = nil.model("gradient_boosting_regressor", max_iter=20, max_leaf_nodes=7, random_state=7)
model.fit([[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 7.0])
print(model.predict([[4.0]]))
```

## Parameters

`max_iter`: boosting iterations. `learning_rate`: update size. `max_leaf_nodes`: tree complexity. Uses histogram gradient boosting. Other arguments follow the native constructor.

## Output

A native `HistGradientBoostingRegressor` instance. Prediction/transform methods return native backend arrays.

## Requirements

Install `nil-sdk[classical]` or the source project with `pip install ".[classical]"`. Only this model's backend is loaded when constructed.

## Limitations

The tiny example demonstrates API use, not model quality. Prepare numeric features and validate on held-out data. NIL passes parameters directly to the backend and adds no per-prediction wrapper. Use native fit/predict or nil.fit/nil.evaluate. nil.train remains neural-specific. This constructor uses histogram gradient boosting; use max_iter rather than n_estimators.
