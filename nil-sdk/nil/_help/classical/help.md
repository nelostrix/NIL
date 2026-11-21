# classical

Train a native estimator/pipeline and receive a consistent result with run records.

## Usage

```python
import nil

model = nil.pipeline(nil.model("linear_regression"), preprocess=nil.preprocessing.numeric())
result = nil.fit(model, [[0.], [1.], [2.]], [1., 3., 5.], validation=([[3.], [4.]], [7., 9.]))
print(result.metrics)
print(result.predict([[5.]]))
```

## Parameters

estimator, features, targets; optional validation=(X,y), task, sample_weight, validation_weight, output_dir, name, native fit parameters.

## Output

ClassicalResult: model, run, metrics, task, warnings and status; predict/evaluate/transform/save methods. Legacy result["model"] and result["metrics"] remain supported.

## Requirements

Install `nil-sdk[classical]` for sklearn operations. Named DataFrame inputs additionally require pandas. Backends load only when called.

## Limitations

Fits mutate the supplied native estimator, as before. Cross-validation/search clone it. Unsupported weights raise rather than being discarded. Classical methods are separate from gradient-based nil.train. Native fitting is not forcibly cancellable or time-limited.

Transformers can be fitted without targets and saved as bundles. Use result.transform(X); no predictive accuracy is fabricated for transformations.
