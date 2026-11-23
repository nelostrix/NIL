# boosting

Explicit native LightGBM and CatBoost CPU models.

## Usage

```python
import nil
model = nil.model('lightgbm_classifier', n_estimators=20)
# Fit with nil.fit(model, X_train, y_train), then nil.save / nil.load.
print(model.get_params()['n_jobs'])
```

## Parameters

Native model parameters; CatBoost defaults to 100 iterations, CPU, one thread, quiet output and no training files. LightGBM defaults to one thread and quiet output. Explicit parameters override these defaults.

## Output

Native estimators, compatible with NIL preprocessing/pipelines and complete trusted bundles.

## Requirements

nil-sdk[lightgbm] or nil-sdk[catboost], explicitly installed in the selected environment.

## Limitations

These are backend integrations, not independent numerical reimplementations. GPU use is explicit and unverified here. Native categorical handling and sample-weight constraints follow the chosen backend. Base NIL does not install them.
