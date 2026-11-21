# truncated_svd

Native `sklearn.decomposition.TruncatedSVD` adapter in unsupervised.

## Usage

```python
import nil

model = nil.model("truncated_svd", n_components=1, random_state=7)
X = [[0., 0.], [0.1, 0.2], [1., 1.], [1.1, 1.2], [0.2, 0.1], [1.2, 1.1]]
print(model.fit_transform(X))
```

## Parameters

n_components and n_iter configure sparse-friendly dimensionality reduction.

## Output

A native TruncatedSVD; fitting/prediction/transform methods retain backend behavior.

## Requirements

Install `nil-sdk[classical]` for sklearn operations. Named DataFrame inputs additionally require pandas. Backends load only when called.

## Limitations

Native model/data constraints apply. This tiny example establishes API usage, not useful quality. Tune only with held-out folds and explicit budgets; some algorithms can require substantial time or memory.
