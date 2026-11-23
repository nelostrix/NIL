# isolation_forest

Native `sklearn.ensemble.IsolationForest` adapter in anomaly_detection.

## Usage

```python
import nil

model = nil.model("isolation_forest", n_estimators=20, n_jobs=1, random_state=7)
X = [[0., 0.], [0.1, 0.2], [1., 1.], [1.1, 1.2], [0.2, 0.1], [1.2, 1.1]]
model.fit(X)
print(model.predict(X[:2]))
```

## Parameters

n_estimators, contamination and max_samples configure anomaly detection.

## Output

A native IsolationForest; fitting/prediction/transform methods retain backend behavior.

## Requirements

Install `nil-sdk[classical]` for sklearn operations. Named DataFrame inputs additionally require pandas. Backends load only when called.

## Limitations

Native model/data constraints apply. This tiny example establishes API usage, not useful quality. Tune only with held-out folds and explicit budgets; some algorithms can require substantial time or memory. Native labels are -1 for outliers and +1 for inliers; they are not ordinary target class IDs.
