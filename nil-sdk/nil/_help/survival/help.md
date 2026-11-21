# survival

Right-censored Cox proportional hazards and Kaplan–Meier through lifelines.

## Usage

```python
import nil
model = nil.survival.kaplan_meier([1, 2, 3, 4], [1, 0, 1, 0])
print(model.predict([0, 1, 2, 3]))
# cox = nil.survival.fit(X_train, durations, observed_events)
# print(nil.survival.evaluate(cox, X_test, test_times, test_events))
```

## Parameters

fit(features,durations,events,penalizer=.1); positive finite durations and 0/1 observed-event flags. Cox predict returns partial hazards; survival_function(X,sorted_times) returns subjects×times probabilities. kaplan_meier(times,events) returns a fitted baseline.

## Output

Serializable cox_ph / kaplan_meier models and held-out Harrell concordance index (larger risk means shorter survival). Bundles record lifelines and relevant pandas versions.

## Requirements

nil-sdk[survival]: explicitly installed lifelines/pandas and classical backend.

## Limitations

Right censoring only. Cox assumes proportional hazards and noninformative censoring; no competing risks, left truncation, time-varying covariates, IPCW Brier scores or survival forests. Kaplan–Meier is an unconditional baseline. Metrics reject datasets without comparable pairs. Ties and fitting follow native lifelines. No medical or causal validity is implied by a numerical test. Use independent held-out subjects and nil.save/load.
