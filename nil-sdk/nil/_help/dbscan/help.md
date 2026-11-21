# dbscan

Native `sklearn.cluster.DBSCAN` adapter in unsupervised.

## Usage

```python
import nil

model = nil.model("dbscan", eps=0.5, min_samples=2)
X = [[0., 0.], [0.1, 0.2], [1., 1.], [1.1, 1.2], [0.2, 0.1], [1.2, 1.1]]
labels = model.fit_predict(X)
print(labels)
```

## Parameters

eps controls neighborhood radius; min_samples sets minimum density.

## Output

A native DBSCAN; fitting/prediction/transform methods retain backend behavior.

## Requirements

Install `nil-sdk[classical]` for sklearn operations. Named DataFrame inputs additionally require pandas. Backends load only when called.

## Limitations

Native model/data constraints apply. This tiny example establishes API usage, not useful quality. Tune only with held-out folds and explicit budgets; some algorithms can require substantial time or memory. These models have no general predict method for unseen samples. Pass explicit cluster_labels when evaluating their training clustering.
