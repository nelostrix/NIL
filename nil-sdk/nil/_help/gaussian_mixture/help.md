# gaussian_mixture

Native `sklearn.mixture.GaussianMixture` adapter in unsupervised.

## Usage

```python
import nil

model = nil.model("gaussian_mixture", n_components=2, random_state=7)
X = [[0., 0.], [0.1, 0.2], [1., 1.], [1.1, 1.2], [0.2, 0.1], [1.2, 1.1]]
model.fit(X)
print(model.predict(X[:2]))
```

## Parameters

n_components, covariance_type and reg_covar configure density modeling.

## Output

A native GaussianMixture; fitting/prediction/transform methods retain backend behavior.

## Requirements

Install `nil-sdk[classical]` for sklearn operations. Named DataFrame inputs additionally require pandas. Backends load only when called.

## Limitations

Native model/data constraints apply. This tiny example establishes API usage, not useful quality. Tune only with held-out folds and explicit budgets; some algorithms can require substantial time or memory.
