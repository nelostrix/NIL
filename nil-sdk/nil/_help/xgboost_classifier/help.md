# xgboost_classifier

Build `xgboost.XGBClassifier` for discrete class labels.

## Usage

```python
import nil

model = nil.model("xgboost_classifier", n_estimators=20, max_depth=2, n_jobs=1, random_state=7)
model.fit([[-2.0], [-1.0], [1.0], [2.0]], [0, 0, 1, 1])
print(model.predict([[-3.0], [3.0]]))
```

## Parameters

`n_estimators`: boosting rounds. `max_depth`: tree complexity. `learning_rate`: update size. `n_jobs`: worker limit. Other arguments follow the native constructor.

## Output

A native `XGBClassifier` instance. Prediction/transform methods return native backend arrays.

## Requirements

Install `nil-sdk[boosting]` or the source project with `pip install ".[boosting]"`. Only this model's backend is loaded when constructed.
For CPU-only installations with compatible wheels, use `nil-sdk[boosting-cpu]`
instead. Choose one XGBoost distribution variant per environment.

## Limitations

The tiny example demonstrates API use, not model quality. Prepare numeric features and validate on held-out data. NIL passes parameters directly to the backend and adds no per-prediction wrapper. Use native fit/predict or nil.fit/nil.evaluate. nil.train remains neural-specific.
