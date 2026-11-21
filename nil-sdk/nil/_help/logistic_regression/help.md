# logistic_regression

Build `sklearn.linear_model.LogisticRegression` for discrete class labels.

## Usage

```python
import nil

model = nil.model("logistic_regression", max_iter=200, random_state=7)
model.fit([[-2.0], [-1.0], [1.0], [2.0]], [0, 0, 1, 1])
print(model.predict([[-3.0], [3.0]]))
```

## Parameters

`max_iter`: solver iteration limit. `C`: inverse regularization strength. `random_state`: seed for solvers that use randomness. Other arguments follow the native constructor.

## Output

A native `LogisticRegression` instance. Prediction/transform methods return native backend arrays.

## Requirements

Install `nil-sdk[classical]` or the source project with `pip install ".[classical]"`. Only this model's backend is loaded when constructed.

## Limitations

The tiny example demonstrates API use, not model quality. Prepare numeric features and validate on held-out data. NIL passes parameters directly to the backend and adds no per-prediction wrapper. Use native fit/predict or nil.fit/nil.evaluate. nil.train remains neural-specific. Despite its name this is classification, not continuous-value regression.
