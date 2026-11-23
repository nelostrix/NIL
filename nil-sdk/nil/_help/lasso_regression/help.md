# lasso_regression

Build `sklearn.linear_model.Lasso` for continuous numeric targets.

## Usage

```python
import nil

model = nil.model("lasso_regression", alpha=0.01, max_iter=1000)
model.fit([[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 7.0])
print(model.predict([[4.0]]))
```

## Parameters

`alpha`: L1 regularization strength. `max_iter`: iteration limit. Other arguments follow the native constructor.

## Output

A native `Lasso` instance. Prediction/transform methods return native backend arrays.

## Requirements

Install `nil-sdk[classical]` or the source project with `pip install ".[classical]"`. Only this model's backend is loaded when constructed.

## Limitations

The tiny example demonstrates API use, not model quality. Prepare numeric features and validate on held-out data. NIL passes parameters directly to the backend and adds no per-prediction wrapper. Use native fit/predict or nil.fit/nil.evaluate. nil.train remains neural-specific.
