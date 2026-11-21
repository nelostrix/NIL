# knn_classifier

Native `sklearn.neighbors.KNeighborsClassifier` adapter in classification.

## Usage

```python
import nil

model = nil.model("knn_classifier", n_neighbors=3)
X = [[0., 0.], [0.1, 0.2], [1., 1.], [1.1, 1.2], [0.2, 0.1], [1.2, 1.1]]
model.fit(X, [0, 0, 1, 1, 0, 1])
print(model.predict(X[:2]))
```

## Parameters

n_neighbors, weights and metric configure neighbor prediction.

## Output

A native KNeighborsClassifier; fitting/prediction/transform methods retain backend behavior.

## Requirements

Install `nil-sdk[classical]` for sklearn operations. Named DataFrame inputs additionally require pandas. Backends load only when called.

## Limitations

Native model/data constraints apply. This tiny example establishes API usage, not useful quality. Tune only with held-out folds and explicit budgets; some algorithms can require substantial time or memory.
