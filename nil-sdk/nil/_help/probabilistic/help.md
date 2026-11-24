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

See docs/SEVEN_WORKFLOWS.md for v1 acceptance and docs/ADVANCED_WORKFLOWS.md for this phase and remaining work.

## Sampling and instrumental variables

metropolis(log_density, initial_chains, target_id=...).sample(draws) provides bounded random-walk sampling; save/load_metropolis preserve state/RNG. Basic split-Rhat/ESS are diagnostics, not convergence proof. instrumental_ate(z,t,y) provides linear 2SLS/HC1; identification assumptions remain caller-owned.
