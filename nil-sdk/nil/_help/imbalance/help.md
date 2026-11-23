# imbalance

Resample only during training and select binary thresholds on held-out validation.

## Usage

```python
import nil

model = nil.imbalance.pipeline(
    nil.model("logistic_regression"),
    preprocess=nil.preprocessing.numeric(),
    resample=nil.imbalance.sampler("over", seed=7))
X = [[float(i)] for i in range(20)]
y = [0] * 15 + [1] * 5
result = nil.fit(model, X, y)
thresholded = nil.imbalance.threshold(result, [[1.], [18.]], [0, 1])
print(thresholded.predict([[2.], [19.]]))
```

## Parameters

`sampler(kind='smote', seed=0, **native_parameters)` accepts smote, over or under.
`pipeline(model, resample=..., preprocess=None)` returns an imbalanced-learn
pipeline. `threshold(model, X_validation, y_validation, metric='f1', candidates=101,
sample_weight=None)` supports f1, balanced_accuracy or accuracy, and 2–10001 grid
points. Both classes need positive validation weight; positive class is classes_[1].

## Output

Native samplers/pipelines. Threshold selection returns a fitted sklearn-protocol
classifier with threshold, selection_score_, selection_metric_ and NIL persistence.
Threshold search uses sorted cumulative weighted counts rather than repeatedly
predicting or fitting at each grid point.

## Requirements

Resampling requires `nil-sdk[imbalance]`. Threshold selection needs only classical.

## Limitations

Use the full resampling pipeline inside CV to keep synthetic rows out of held-out
folds. Prediction never resamples. NIL rejects sample_weight forwarding through
resampling because row counts change; choose class weighting or a sampler with an
explicit weight strategy instead. SMOTE needs enough minority neighbors and
numeric distances; it is not a categorical-data synthesis method. Threshold
selection needs validation data separate from training and final test data.
