"""Shared contracts; importing this module loads no compute engine."""
from numbers import Integral

from nil.runtime import require


def unwrap(model):
    return model.model if getattr(model, "_nil_classical_result", False) or getattr(model, "_nil_rl_result", False) else model


def is_rl(model):
    model = unwrap(model)
    if getattr(model,'_nil_rl_policy',False): return True
    return callable(getattr(model, 'learn', None)) and callable(getattr(model, 'predict', None)) and any(
        cls.__module__.startswith('stable_baselines3.') for cls in type(model).__mro__)


def is_classical(model):
    model = unwrap(model)
    return callable(getattr(model, "get_params", None)) and callable(getattr(model, "fit", None))


def task_for(model, task=None):
    if task is not None:
        if task not in {"classification", "regression", "clustering", "anomaly_detection", "transformation"}:
            raise ValueError("Choose classification, regression, clustering, anomaly_detection or transformation")
        return task
    base = require("sklearn.base", "classical")
    model = unwrap(model)
    if hasattr(model, "steps"):
        return task_for(model.steps[-1][1])
    if base.is_classifier(model):
        return "classification"
    if base.is_regressor(model):
        return "regression"
    if getattr(base, "is_outlier_detector", lambda value: False)(model) or getattr(model, "_estimator_type", None) == "outlier_detector":
        return "anomaly_detection"
    if getattr(base, "is_clusterer", lambda value: False)(model) or getattr(model, "_estimator_type", None) == "clusterer" or hasattr(model, "labels_") or type(model).__name__ in {"KMeans", "DBSCAN", "AgglomerativeClustering", "GaussianMixture"}:
        return "clustering"
    if callable(getattr(model, "transform", None)):
        return "transformation"
    raise ValueError("Cannot infer this estimator's task; pass task= explicitly")


def weights_for(weights, targets):
    if weights is None:
        return None
    np = require("numpy", "classical")
    values = np.asarray(weights, dtype=float)
    count = targets.shape[0] if hasattr(targets, "shape") else len(targets)
    if values.ndim != 1 or len(values) != count:
        raise ValueError("sample_weight must have one value per sample")
    if not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any():
        raise ValueError("sample_weight must be finite, nonnegative and have positive total weight")
    return values


def weighted_fit_params(estimator, sample_weight=None, fit_params=None):
    params = dict(fit_params or {})
    if sample_weight is not None:
        prefix = ""
        while hasattr(estimator, "steps"):
            if any(callable(getattr(step, "fit_resample", None)) for _, step in estimator.steps):
                raise ValueError("sample_weight cannot be forwarded through resampling; use class weights or a weighted sampler explicitly")
            prefix += estimator.steps[-1][0] + "__"
            estimator = estimator.steps[-1][1]
        key = prefix + "sample_weight"
        if key in params:
            raise ValueError(f"Pass weights once, not both sample_weight and {key}")
        params[key] = sample_weight
    return params


def warning_records(records):
    return [{"category": item.category.__name__, "message": str(item.message)} for item in records]


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
