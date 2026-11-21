# uncertainty

Split-conformal regression intervals and single-output classification label sets.

## Usage

```python
import nil

model = nil.model("linear_regression").fit([[0.], [1.], [2.]], [0., 2., 4.])
X_cal = [[float(i)] for i in range(10, 30)]
y_cal = [2 * x[0] + (-1 if i % 2 else 1) for i, x in enumerate(X_cal)]
conformal = nil.uncertainty.calibrate(model, X_cal, y_cal, alpha=0.1)
print(conformal.interval([[31.]]))
print(nil.uncertainty.evaluate(conformal, [[31.]], [62.]))
nil.save(conformal, "./conformal-v1")
```

## Parameters

`calibrate(model, features, targets, alpha=.1)` expects a previously fitted model
and separate calibration data. Classifiers must supply normalized predict_proba
and known labels. `evaluate` requires separate final evaluation rows.

## Output

Regression `interval(X)` returns lower/upper columns. Classification
`prediction_sets(X)` returns a boolean matrix in `classes_` order. Evaluation
reports empirical coverage plus mean interval width or set size. Wrapped models
support native prediction, cloning and NIL persistence.

## Requirements

`nil-sdk[classical]`. Uses a finite-sample ceiling quantile, with no model refit
during calibration. Undersized calibration sets for the requested alpha fail.

## Limitations

Calibration must be independent of fitting/model selection. Guarantees rely on
exchangeability; distribution shift, temporal dependence and reused calibration
data can invalidate them. Intervals are symmetric and marginal, not conditional
coverage guarantees. Classification uses nonconformity scores
`1 - probability_of_true_label` with a pooled quantile (LAC), not classwise
coverage. Empty label sets are possible. No sample weights, multioutput, censored
data or time-series coverage claim is made. Refitting invalidates calibration;
run calibrate again afterwards. Do not mutate the underlying calibrated model.

Method background: [Angelopoulos and Bates, A Gentle Introduction to Conformal
Prediction](https://arxiv.org/abs/2107.07511).


## Additional calibrated regression contracts

All methods require an independently fitted frozen model and disjoint held-out
calibration labels. New wrappers use nil.save/load and never refit the model.
`evaluate(calibrated,X,y,**interval_options)` reports empirical coverage and width;
for vector outputs coverage means all coordinates simultaneously.

- `calibrate_multioutput(model,X,Y,alpha=.1,scales=None)` uses the maximum absolute
  scaled residual per vector. Scales must be positive and fixed independently of
  calibration labels. The guarantee is simultaneous vector coverage for
  exchangeable calibration/test vectors, not separate unadjusted coordinate
  intervals. interval(X) returns rows×outputs×2.
- `calibrate_groups(model,X,y,groups,alpha=.1)` uses a separate finite-sample
  quantile for each prespecified group. interval(X,groups) refuses unseen groups
  and calibration cells too small for alpha. Exchangeability within each fixed
  group supports group-conditional coverage; it does not imply arbitrary
  pointwise conditional coverage or validity for groups chosen from residuals.
- `calibrate_weighted(model,X,y,likelihood_ratios,alpha=.1)` handles known
  target/source covariate density ratios with unchanged Y|X. The caller supplies
  finite positive calibration ratios and interval(X,test_likelihood_ratios).
  The test point's mass at infinity is included: insufficient calibration mass
  returns an unbounded interval, never a falsely finite radius. Arbitrary sample
  weights or estimated ratios do not inherit the exact known-ratio guarantee.
- `calibrate_blocks(model,X,y,block_size,alpha=.1)` calibrates maxima over complete
  contiguous nonoverlapping blocks. interval(X) requires exactly one next block.
  Simultaneous coverage requires exchangeability of whole blocks; dependence
  within a block is allowed. Arbitrary time-series drift/dependence does not
  satisfy that assumption, and no universal temporal guarantee is claimed.

Primary weighted-conformal reference: Tibshirani et al., *Conformal Prediction
Under Covariate Shift* (2019), https://arxiv.org/abs/1904.06019 . Test evidence
includes finite-rank boundary calculations, simultaneous-vector/group/block
contracts, and a seeded held-out covariate-shift coverage check; a particular
empirical coverage result is not a guarantee for a user's data.
