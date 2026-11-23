# sgd_regressor

Native `sklearn.linear_model.SGDRegressor` adapter in regression.

## Usage

```python
import nil

model = nil.model("sgd_regressor", max_iter=1000, random_state=7)
X = [[0., 0.], [0.1, 0.2], [1., 1.], [1.1, 1.2], [0.2, 0.1], [1.2, 1.1]]
model.fit(X, [0., 0.2, 2., 2.2, 0.3, 2.3])
print(model.predict(X[:2]))
```

## Parameters

loss, penalty, alpha, max_iter and learning_rate follow native SGD.

## Output

A native SGDRegressor; fitting/prediction/transform methods retain backend behavior.

## Requirements

Install `nil-sdk[classical]` for sklearn operations. Named DataFrame inputs additionally require pandas. Backends load only when called.

## Limitations

Native model/data constraints apply. This tiny example establishes API usage, not useful quality. Tune only with held-out folds and explicit budgets; some algorithms can require substantial time or memory.
