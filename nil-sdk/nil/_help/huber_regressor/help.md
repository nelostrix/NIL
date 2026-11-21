# huber_regressor

Native `sklearn.linear_model.HuberRegressor` adapter in regression.

## Usage

```python
import nil

model = nil.model("huber_regressor", epsilon=1.35)
X = [[0., 0.], [0.1, 0.2], [1., 1.], [1.1, 1.2], [0.2, 0.1], [1.2, 1.1]]
model.fit(X, [0., 0.2, 2., 2.2, 0.3, 2.3])
print(model.predict(X[:2]))
```

## Parameters

epsilon sets the outlier cutoff; alpha controls regularization.

## Output

A native HuberRegressor; fitting/prediction/transform methods retain backend behavior.

## Requirements

Install `nil-sdk[classical]` for sklearn operations. Named DataFrame inputs additionally require pandas. Backends load only when called.

## Limitations

Native model/data constraints apply. This tiny example establishes API usage, not useful quality. Tune only with held-out folds and explicit budgets; some algorithms can require substantial time or memory.
