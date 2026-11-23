# pca

Build `sklearn.decomposition.PCA` for unlabeled numeric observations.

## Usage

```python
import nil

model = nil.model("pca", n_components=1)
projection = model.fit_transform([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
print(projection)
```

## Parameters

`n_components`: number of retained components; must fit the data dimensions. Other arguments follow the native constructor.

## Output

A native `PCA` instance. Prediction/transform methods return native backend arrays.

## Requirements

Install `nil-sdk[classical]` or the source project with `pip install ".[classical]"`. Only this model's backend is loaded when constructed.

## Limitations

PCA transforms features; it does not predict target labels. Scale features when appropriate and fit transformations on training data only.
