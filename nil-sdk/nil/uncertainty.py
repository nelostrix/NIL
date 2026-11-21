"""Split-conformal intervals/sets using independent held-out calibration data."""

import math

from nil._classical_common import task_for, unwrap
from nil.runtime import require


def _quantile(scores, alpha):
    np = require("numpy", "classical")
    if isinstance(alpha, bool) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be nonempty and finite")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError("Calibration sample is too small for this alpha; collect more calibration data")
    return float(np.partition(scores, rank - 1)[rank - 1])


def _probabilities(values, classes):
    np = require("numpy", "classical")
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != classes or not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Expected a finite normalized probability matrix")
    return probabilities


def calibrate(model, features, targets, *, alpha=0.1):
    """Never refits the model. Calibration rows must be independent of its training."""
    np = require("numpy", "classical")
    validation = require("sklearn.utils.validation", "classical")
    model = unwrap(model)
    validation.check_is_fitted(model)
    validation.check_consistent_length(features, targets)
    actual = np.asarray(targets)
    if actual.ndim != 1:
        raise ValueError("Conformal calibration currently requires one output per sample")
    task = task_for(model)
    if task == "regression":
        from nil._classical_estimators import ConformalRegressor
        result = ConformalRegressor(model, alpha=alpha)
        predicted = np.asarray(model.predict(features))
        if predicted.shape != actual.shape:
            raise ValueError("Expected one regression prediction per row")
        result.radius_ = _quantile(np.abs(actual - predicted), alpha)
    elif task == "classification":
        from nil._classical_estimators import ConformalClassifier
        if not hasattr(model, "predict_proba"):
            raise ValueError("Conformal classification requires predict_proba")
        classes = np.asarray(model.classes_)
        if classes.ndim != 1 or not np.isin(actual, classes).all():
            raise ValueError("Calibration labels must belong to the model's single-output classes")
        result = ConformalClassifier(model, alpha=alpha)
        result.classes_ = classes
        probabilities = _probabilities(model.predict_proba(features), len(classes))
        indices = np.argmax(actual[:, None] == classes[None, :], axis=1)
        result.cutoff_ = _quantile(1 - probabilities[np.arange(len(actual)), indices], alpha)
    else:
        raise ValueError("Conformal calibration supports regression and classification")
    result.estimator_ = model
    result.calibration_count_ = len(actual)
    return result


def evaluate(model, features, targets, **interval_options):
    np = require("numpy", "classical")
    require("sklearn.utils.validation", "classical").check_consistent_length(features, targets)
    actual = np.asarray(targets)
    if actual.ndim not in {1,2} or not len(actual):
        raise ValueError("Uncertainty evaluation needs nonempty regression vectors or classification labels")
    if hasattr(model, "interval"):
        intervals = model.interval(features, **interval_options)
        if intervals.shape != actual.shape + (2,) or not np.isfinite(np.asarray(actual,dtype=float)).all():
            raise ValueError('Interval/target shape mismatch or invalid targets')
        covered = (actual >= intervals[..., 0]) & (actual <= intervals[..., 1])
        if actual.ndim==2: covered=covered.all(axis=1)
        return {"count": len(actual), "coverage": float(np.mean(covered)),
                "mean_width": float(np.mean(intervals[..., 1] - intervals[..., 0]))}
    if actual.ndim != 1: raise ValueError("Classification sets require one label per row")
    classes = model.classes_
    if not np.isin(actual, classes).all():
        raise ValueError("Evaluation targets contain unknown classes")
    sets = model.prediction_sets(features)
    indices = np.argmax(actual[:, None] == classes[None, :], axis=1)
    return {"count": len(actual), "coverage": float(np.mean(sets[np.arange(len(actual)), indices])),
            "mean_set_size": float(np.mean(sets.sum(axis=1)))}


def _regression_scores(model, features, targets, *, multi=False):
    np=require('numpy','classical'); validation=require('sklearn.utils.validation','classical')
    model=unwrap(model); validation.check_is_fitted(model); validation.check_consistent_length(features,targets)
    actual=np.asarray(targets,dtype=float); predicted=np.asarray(model.predict(features),dtype=float)
    if actual.shape!=predicted.shape or actual.ndim!=(2 if multi else 1) or not len(actual) or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Expected matching finite regression targets/predictions')
    return model,np.abs(actual-predicted)


def calibrate_multioutput(model, features, targets, *, alpha=.1, scales=None):
    """Simultaneous coverage of an entire exchangeable output vector."""
    np=require('numpy','classical'); model,scores=_regression_scores(model,features,targets,multi=True)
    scale=np.ones(scores.shape[1]) if scales is None else np.asarray(scales,dtype=float)
    if scale.shape!=(scores.shape[1],) or not np.isfinite(scale).all() or (scale<=0).any(): raise ValueError('Provide fixed positive output scales, independent of calibration labels')
    from nil._uncertainty import MultioutputIntervals
    result=MultioutputIntervals(model,alpha=alpha); result.estimator_=model; result.scales_=scale
    result.radius_=_quantile((scores/scale).max(axis=1),alpha)
    return result


def calibrate_groups(model, features, targets, groups, *, alpha=.1):
    """Mondrian intervals within prespecified exchangeable groups, not arbitrary pointwise coverage."""
    np=require('numpy','classical'); model,scores=_regression_scores(model,features,targets)
    from nil._uncertainty import GroupIntervals,group_keys
    keys=group_keys(groups,len(scores)); result=GroupIntervals(model,alpha=alpha); result.estimator_=model
    cells={}
    for key,score in zip(keys,scores): cells.setdefault(key,[]).append(score)
    result.radii_={key:_quantile(values,alpha) for key,values in cells.items()}
    return result


def calibrate_weighted(model, features, targets, likelihood_ratios, *, alpha=.1):
    """Covariate-shift conformal calibration with known target/source density ratios."""
    np=require('numpy','classical'); model,scores=_regression_scores(model,features,targets)
    from nil._uncertainty import WeightedIntervals,positive_weights
    if isinstance(alpha,bool) or not 0<alpha<1: raise ValueError('alpha must be between zero and one')
    weights=positive_weights(likelihood_ratios,len(scores))
    order=np.argsort(scores,kind='stable'); result=WeightedIntervals(model,alpha=alpha); result.estimator_=model
    result.scores_,result.weights_=scores[order],weights[order]
    if not np.isfinite(weights.sum()): raise ValueError('Likelihood ratios overflow their total')
    return result


def calibrate_blocks(model, features, targets, *, block_size, alpha=.1):
    """Simultaneous next-block intervals under whole-block exchangeability."""
    from nil._classical_common import positive_int
    from nil._uncertainty import BlockIntervals
    positive_int(block_size,'block_size'); model,scores=_regression_scores(model,features,targets)
    if len(scores)%block_size: raise ValueError('Calibration must contain complete nonoverlapping blocks')
    result=BlockIntervals(model,block_size=block_size,alpha=alpha); result.estimator_=model
    result.radius_=_quantile(scores.reshape(-1,block_size).max(axis=1),alpha)
    return result
