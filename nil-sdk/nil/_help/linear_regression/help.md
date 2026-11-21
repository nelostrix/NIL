# linear_regression

Build `sklearn.linear_model.LinearRegression` for continuous numeric targets.

## Usage

```python
import nil

model = nil.model("linear_regression", fit_intercept=True)
model.fit([[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 7.0])
print(model.predict([[4.0]]))
```

## Parameters

`fit_intercept`: include a constant offset in the model. Other arguments follow the native constructor.

## Output

A native `LinearRegression` instance. Prediction/transform methods return native backend arrays.

## Requirements

Install `nil-sdk[classical]` or the source project with `pip install ".[classical]"`. Only this model's backend is loaded when constructed.

## Limitations

The tiny example demonstrates API use, not model quality. Prepare numeric features and validate on held-out data. NIL passes parameters directly to the backend and adds no per-prediction wrapper. Use native fit/predict or nil.fit/nil.evaluate. nil.train remains neural-specific.
