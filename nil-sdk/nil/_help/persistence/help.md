# persistence

Save the complete fitted classical pipeline, feature schema and dependency versions.

## Usage

```python
import nil

result = nil.fit(nil.model("linear_regression"), [[0.], [1.], [2.]], [1., 3., 5.])
result.save("./saved-linear")
loaded = nil.load("./saved-linear", trusted=True)
print(loaded.predict([[3.]]))
```

## Parameters

save(model_or_result, new_directory); load(path, trusted=True, allow_version_mismatch=False).

## Output

A new local bundle directory; loading returns the native fitted pipeline/estimator.

## Requirements

Install `nil-sdk[classical]` for sklearn operations. Named DataFrame inputs additionally require pandas. Backends load only when called.

## Limitations

Joblib artifacts can execute Python during loading: load only your own/trusted artifacts. SHA-256 is integrity, not authenticity. Exact package versions are checked by default. Custom estimator code must remain importable. Neural models continue to use checkpoints.
