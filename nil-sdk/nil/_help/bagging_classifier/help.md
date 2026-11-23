# bagging_classifier

Build `sklearn.ensemble.BaggingClassifier` for discrete class labels.

## Usage

```python
import nil

model = nil.model("bagging_classifier", n_estimators=10, n_jobs=1, random_state=7)
model.fit([[-2.0], [-1.0], [1.0], [2.0]], [0, 0, 1, 1])
print(model.predict([[-3.0], [3.0]]))
```

## Parameters

`n_estimators`: number of learners. `n_jobs`: worker limit. `random_state`: seed. Other arguments follow the native constructor.

## Output

A native `BaggingClassifier` instance. Prediction/transform methods return native backend arrays.

## Requirements

Install `nil-sdk[classical]` or the source project with `pip install ".[classical]"`. Only this model's backend is loaded when constructed.

## Limitations

The tiny example demonstrates API use, not model quality. Prepare numeric features and validate on held-out data. NIL passes parameters directly to the backend and adds no per-prediction wrapper. Use native fit/predict or nil.fit/nil.evaluate. nil.train remains neural-specific.
