"""Fold-local resampling and held-out binary threshold selection."""

from nil._classical_common import positive_int, unwrap, weights_for
from nil.runtime import require


def sampler(kind="smote", *, seed=0, **parameters):
    if kind in {"smote", "over"}:
        module = require("imblearn.over_sampling", "imbalance")
        cls = module.SMOTE if kind == "smote" else module.RandomOverSampler
    elif kind == "under":
        cls = require("imblearn.under_sampling", "imbalance").RandomUnderSampler
    else:
        raise ValueError("Choose smote, over or under resampling")
    return cls(random_state=seed, **parameters)


def pipeline(model, *, resample, preprocess=None):
    Pipeline = require("imblearn.pipeline", "imbalance").Pipeline
    steps = [("preprocess", preprocess)] if preprocess is not None else []
    steps.extend([("resample", resample), ("model", unwrap(model))])
    return Pipeline(steps)


def threshold(model, features, targets, *, metric="f1", candidates=101, sample_weight=None):
    """Select on validation data, never final test data; positive class is classes_[1]."""
    np = require("numpy", "classical")
    validation = require("sklearn.utils.validation", "classical")
    model = unwrap(model)
    validation.check_is_fitted(model)
    validation.check_consistent_length(features, targets)
    classes = np.asarray(model.classes_)
    actual = np.asarray(targets)
    if classes.ndim != 1 or len(classes) != 2 or actual.ndim != 1 or not len(actual) or not np.isin(actual, classes).all():
        raise ValueError("Threshold selection requires known binary labels")
    if metric not in {"f1", "balanced_accuracy", "accuracy"}:
        raise ValueError("Choose f1, balanced_accuracy or accuracy")
    positive_int(candidates, "candidates")
    if candidates < 2 or candidates > 10001:
        raise ValueError("candidates must be between 2 and 10001")
    weight = weights_for(sample_weight, actual)
    from nil.uncertainty import _probabilities
    probabilities = _probabilities(model.predict_proba(features), 2)[:, 1]
    binary = actual == classes[1]
    weight = np.ones(len(actual)) if weight is None else weight
    order = np.argsort(probabilities, kind="stable")
    positive = np.concatenate(([0.], np.cumsum(weight[order] * binary[order])))
    negative = np.concatenate(([0.], np.cumsum(weight[order] * ~binary[order])))
    if positive[-1] <= 0 or negative[-1] <= 0:
        raise ValueError("Both classes need positive validation weight for threshold selection")
    grid = np.linspace(0, 1, candidates)
    positions = np.searchsorted(probabilities[order], grid, side="left")
    tp, tn = positive[-1] - positive[positions], negative[positions]
    fp, fn = negative[-1] - tn, positive[positions]
    if metric == "f1":
        denominator = 2 * tp + fp + fn
        scores = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    elif metric == "balanced_accuracy":
        scores = .5 * (tp / positive[-1] + tn / negative[-1])
    else:
        scores = (tp + tn) / (positive[-1] + negative[-1])
    index = int(np.argmax(scores))
    from nil._classical_estimators import ThresholdClassifier
    result = ThresholdClassifier(model, threshold=float(grid[index]))
    result.estimator_, result.classes_ = model, classes
    result.selection_score_, result.selection_metric_ = float(scores[index]), metric
    return result
