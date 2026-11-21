# select_from_model

Native `sklearn.feature_selection.SelectFromModel` adapter in feature_selection.

## Usage

```python
import nil

model = nil.model("select_from_model", estimator=nil.model("logistic_regression"))
X = [[0., 0.], [0.1, 0.2], [1., 1.], [1.1, 1.2], [0.2, 0.1], [1.2, 1.1]]
transformed = model.fit_transform(X, [0, 0, 1, 1, 0, 1])
print(transformed.shape)
```

## Parameters

estimator supplies fitted coefficients/importances; threshold controls selection.

## Output

A native SelectFromModel; fitting/prediction/transform methods retain backend behavior.

## Requirements

Install `nil-sdk[classical]` for sklearn operations. Named DataFrame inputs additionally require pandas. Backends load only when called.

## Limitations

Native model/data constraints apply. This tiny example establishes API usage, not useful quality. Tune only with held-out folds and explicit budgets; some algorithms can require substantial time or memory. Fit learned selectors inside nil.pipeline(select=...) so CV does not leak feature-selection information.
