# variance_threshold

Native `sklearn.feature_selection.VarianceThreshold` adapter in feature_selection.

## Usage

```python
import nil

model = nil.model("variance_threshold", threshold=0.0)
X = [[0., 0.], [0.1, 0.2], [1., 1.], [1.1, 1.2], [0.2, 0.1], [1.2, 1.1]]
transformed = model.fit_transform(X, [0, 0, 1, 1, 0, 1])
print(transformed.shape)
```

## Parameters

threshold removes features below the chosen variance.

## Output

A native VarianceThreshold; fitting/prediction/transform methods retain backend behavior.

## Requirements

Install `nil-sdk[classical]` for sklearn operations. Named DataFrame inputs additionally require pandas. Backends load only when called.

## Limitations

Native model/data constraints apply. This tiny example establishes API usage, not useful quality. Tune only with held-out folds and explicit budgets; some algorithms can require substantial time or memory. Fit learned selectors inside nil.pipeline(select=...) so CV does not leak feature-selection information.
