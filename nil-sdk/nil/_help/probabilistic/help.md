# probabilistic

## Usage

```python
import nil
posterior = nil.probabilistic.BetaBernoulli().update([1,1,0])
print(posterior.predict())
# effect = nil.probabilistic.causal_ate(X, treatment, outcome)
```

## Parameters

Beta prior alpha/beta; Gaussian prior precision and known observation variance; causal folds, seed, overlap threshold, confidence and explicit nuisance estimators.

## Output

Exact conjugate mean/variance, Gaussian predictive mean/std, JSON posterior snapshots, or AIPW estimate/asymptotic interval and assumptions.

## Requirements

Beta uses standard library; BayesianLinear needs NumPy; causal_ate needs the classical extra.

## Limitations

No general probabilistic-programming or causal-discovery engine. ATE assumes consistency, positivity, independent rows and no unmeasured confounding; these cannot be established by this function.

See docs/SEVEN_WORKFLOWS.md and the bundled workflow examples for complete protocols and acceptance.
