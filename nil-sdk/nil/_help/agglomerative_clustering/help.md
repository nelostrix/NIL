# agglomerative_clustering

Native `sklearn.cluster.AgglomerativeClustering` adapter in unsupervised.

## Usage

```python
import nil

model = nil.model("agglomerative_clustering", n_clusters=2)
X = [[0., 0.], [0.1, 0.2], [1., 1.], [1.1, 1.2], [0.2, 0.1], [1.2, 1.1]]
labels = model.fit_predict(X)
print(labels)
```

## Parameters

n_clusters, linkage and metric configure hierarchical clustering.

## Output

A native AgglomerativeClustering; fitting/prediction/transform methods retain backend behavior.

## Requirements

Install `nil-sdk[classical]` for sklearn operations. Named DataFrame inputs additionally require pandas. Backends load only when called.

## Limitations

Native model/data constraints apply. This tiny example establishes API usage, not useful quality. Tune only with held-out folds and explicit budgets; some algorithms can require substantial time or memory. These models have no general predict method for unseen samples. Pass explicit cluster_labels when evaluating their training clustering.
