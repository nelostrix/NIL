# kmeans

Build `sklearn.cluster.KMeans` for unlabeled numeric observations.

## Usage

```python
import nil

model = nil.model("kmeans", n_clusters=2, n_init=10, random_state=7)
model.fit([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
print(model.predict([[0.2, 0.2]]))
```

## Parameters

`n_clusters`: cluster count. `n_init`: initialization runs. `random_state`: seed. Other arguments follow the native constructor.

## Output

A native `KMeans` instance. Prediction/transform methods return native backend arrays.

## Requirements

Install `nil-sdk[classical]` or the source project with `pip install ".[classical]"`. Only this model's backend is loaded when constructed.

## Limitations

Cluster labels are arbitrary identifiers, not known classes. Scale features when appropriate and choose cluster count explicitly.
