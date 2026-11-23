"""Explicit model inspection, fold-safe calibration and bounded learning curves."""
import inspect
from numbers import Integral, Real

from nil._classical_common import positive_int, unwrap, weighted_fit_params, weights_for
from nil.runtime import require


def describe(model):
    result = model
    model = unwrap(model)
    report = {"estimator": f"{type(model).__module__}.{type(model).__name__}",
              "feature_count": int(model.n_features_in_) if hasattr(model, "n_features_in_") else None}
    if hasattr(model, "feature_names_in_"):
        report["feature_names"] = model.feature_names_in_.tolist()
    if hasattr(model, "classes_"):
        classes = model.classes_
        report["classes"] = [value.tolist() for value in classes] if isinstance(classes, list) else classes.tolist()
    if getattr(result, "_nil_classical_result", False):
        fitted = result.result if hasattr(result, "result") else result
        report["warnings"] = fitted.warnings
    return report


def feature_importance(model, features=None, targets=None, *, method="auto", scoring=None,
                       repeats=5, seed=0, n_jobs=1, sample_weight=None, max_samples=1.0):
    model = unwrap(model)
    np = require("numpy", "classical")
    if method not in {"auto", "native", "coefficients", "permutation"}:
        raise ValueError("Choose auto, native, coefficients or permutation")
    final = model.steps[-1][1] if hasattr(model, "steps") else model
    if method == "auto":
        method = "native" if hasattr(final, "feature_importances_") else "coefficients" if hasattr(final, "coef_") else "permutation"
    if method == "permutation":
        if features is None or targets is None:
            raise ValueError("Permutation importance requires held-out features and targets")
        positive_int(repeats, "repeats")
        metrics = require("sklearn.inspection", "classical")
        weight = weights_for(sample_weight, targets)
        result = metrics.permutation_importance(model, features, targets, scoring=scoring,
            n_repeats=repeats, random_state=seed, n_jobs=n_jobs, sample_weight=weight, max_samples=max_samples)
        names = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else [f"x{i}" for i in range(len(result.importances_mean))]
        return {"method": method, "features": names, "values": result.importances_mean.tolist(), "std": result.importances_std.tolist()}
    attribute = "feature_importances_" if method == "native" else "coef_"
    if not hasattr(final, attribute):
        raise ValueError(f"Estimator does not provide {attribute}; use permutation importance")
    raw = np.asarray(getattr(final, attribute))
    values = raw if raw.ndim == 1 else np.mean(np.abs(raw), axis=0)
    if method == "coefficients":
        values = np.abs(values)
    names = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
    if hasattr(model, "steps") and len(model.steps) > 1:
        try:
            names = list(model[:-1].get_feature_names_out())
        except (AttributeError, ValueError):
            names = None
    if names is None or len(names) != len(values):
        names = [f"x{i}" for i in range(len(values))]
    report = {"method": method, "features": names, "values": values.tolist()}
    if method == "coefficients":
        report["coefficients"] = raw.tolist()
    return report


def calibrate(estimator, features, targets, *, method="sigmoid", cv=3, groups=None,
              sample_weight=None, seed=0, n_jobs=1, max_fits=100, output_dir="./nil_runs"):
    """Keep preprocessing inside the base estimator so calibration folds refit it."""
    from nil.validation import _plan
    from nil.classical import fit
    estimator = unwrap(estimator)
    plan = _plan(estimator, features, targets, cv, groups, None, seed, 0, max_fits)
    params = {}
    weight = weights_for(sample_weight, targets)
    if weight is not None:
        final = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
        signature = inspect.signature(final.fit).parameters
        if "sample_weight" not in signature and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.values()):
            raise ValueError("The base estimator cannot accept calibration training weights")
        if hasattr(estimator, "steps"):
            params = weighted_fit_params(estimator, weight)
    calibration = require("sklearn.calibration", "classical").CalibratedClassifierCV
    clone = require("sklearn.base", "classical").clone
    model = calibration(estimator=clone(estimator), method=method, cv=plan, n_jobs=n_jobs)
    return fit(model, features, targets, sample_weight=weight, output_dir=output_dir,
               name="calibration", **params)


def learning_curve(estimator, features, targets, *, train_sizes=(0.3, 0.6, 1.0), cv=5,
                   scoring=None, method=None, groups=None, sample_weight=None,
                   seed=0, gap=0, n_jobs=1, max_fits=100, shuffle=True):
    from nil.validation import _plan, cross_validate
    estimator = unwrap(estimator)
    plan = _plan(estimator, features, targets, cv, groups, method, seed, gap, max_fits)
    np = require("numpy", "classical")
    minimum = min(len(train) for train, _ in plan)
    sizes = []
    for value in train_sizes:
        if isinstance(value, Real) and not isinstance(value, Integral):
            if not 0 < value <= 1:
                raise ValueError("Fractional train_sizes must be in (0, 1]")
            size = max(1, int(value * minimum))
        else:
            size = positive_int(value, "train_size")
        if size > minimum:
            raise ValueError("train_size exceeds the smallest training fold")
        sizes.append(size)
    sizes = sorted(set(sizes))
    if not sizes or len(sizes) * len(plan) > max_fits:
        raise ValueError("Learning curve is empty or exceeds max_fits")
    chronological = method == "time" or type(cv).__name__ == "TimeSeriesSplit"
    classification = require("sklearn.base", "classical").is_classifier(estimator)
    if classification:
        classification = require("sklearn.utils.multiclass", "classical").type_of_target(targets) in {"binary", "multiclass"}
    selection = require("sklearn.model_selection", "classical")
    take = require("sklearn.utils", "classical")._safe_indexing
    outputs = []
    for size in sizes:
        subsets = []
        for index, (train, test) in enumerate(plan):
            if size == len(train):
                subset = train
            elif shuffle and not chronological:
                subset, _ = selection.train_test_split(train, train_size=size, random_state=seed + index,
                    stratify=take(targets, train) if classification else None)
            else:
                subset = train[:size]
            subsets.append((np.asarray(subset), test))
        result = cross_validate(estimator, features, targets, cv=subsets, groups=groups,
            sample_weight=sample_weight, scoring=scoring, n_jobs=n_jobs, max_fits=max_fits, return_train_score=True)
        outputs.append({"train_size": size, "test_scores": result.scores,
                        "train_scores": result.train_scores, "summary": result.summary,
                        "warnings": [notice for fold in result.folds for notice in fold["warnings"]]})
    return {"train_sizes": sizes, "points": outputs, "fit_count": len(sizes) * len(plan)}


def partial_dependence(model, data, *, columns, kind="average", grid_resolution=30,
                       max_samples=200, max_evaluations=100000, seed=0):
    """Compute PDP/ICE numerically on a bounded held-out sample, without plotting."""
    np = require("numpy", "classical")
    positive_int(grid_resolution, "grid_resolution")
    positive_int(max_samples, "max_samples")
    positive_int(max_evaluations, "max_evaluations")
    if isinstance(columns, str):
        columns = [columns]
    columns = list(columns)
    if not 1 <= len(columns) <= 2 or kind not in {"average", "individual", "both"}:
        raise ValueError("Select one or two columns and average, individual or both")
    count = data.shape[0] if hasattr(data, "shape") else len(data)
    count = min(count, max_samples)
    if not count or count * grid_resolution ** len(columns) > max_evaluations:
        raise ValueError("Partial dependence is empty or exceeds max_evaluations")
    take = require("sklearn.utils", "classical")._safe_indexing
    total = data.shape[0] if hasattr(data, "shape") else len(data)
    indices = np.random.RandomState(seed).choice(total, size=count, replace=False)
    result = require("sklearn.inspection", "classical").partial_dependence(
        unwrap(model), take(data, indices), features=columns, kind=kind,
        method="brute", grid_resolution=grid_resolution)
    grids = result["grid_values"] if "grid_values" in result else result["values"]
    return {"sample_count": count, "row_indices": indices.tolist(),
            "grid_values": [part.tolist() for part in grids],
            **{key: result[key].tolist() for key in ("average", "individual") if key in result}}


def compare(models, features, targets, *, cv=5, scoring=None, groups=None,
            sample_weight=None, seed=0, max_fits=100):
    """Compare cloned candidates on identical CV folds. No final test-set selection."""
    from nil.validation import _plan, _scorers, cross_validate
    if not isinstance(models, dict) or not models:
        raise ValueError("Provide a nonempty mapping of candidate names to estimators")
    first = unwrap(next(iter(models.values())))
    plan = _plan(first, features, targets, cv, groups, None, seed, 0, max_fits)
    if len(models) * len(plan) > max_fits:
        raise ValueError("Candidate comparison exceeds max_fits")
    scorers = _scorers(first, scoring, targets)
    if len(scorers) != 1:
        raise ValueError("Candidate comparison needs one ranking metric")
    metric = next(iter(scorers))
    records = {}
    for name, model in models.items():
        result = cross_validate(model, features, targets, cv=plan, groups=groups,
            scoring=scorers, sample_weight=sample_weight, max_fits=max_fits)
        records[name] = {"summary": result.summary, "scores": result.scores,
                         "fit_time": sum(item["fit_time"] for item in result.folds)}
    ranking = sorted(records, key=lambda name: records[name]["summary"][metric]["mean"], reverse=True)
    return {"metric": metric, "ranking": ranking, "results": records,
            "fit_count": len(models) * len(plan)}


def shap_values(model, features, background, *, permutations=8, max_rows=32,
                max_background=32, max_features=64, max_evaluations=100000,
                max_dense_bytes=32000000, class_index=None, seed=0):
    """Bounded interventional permutation Shapley values, not TreeSHAP."""
    from nil._explanations import permutation_shap
    return permutation_shap(model, features, background, permutations=permutations,
        max_rows=max_rows, max_background=max_background, max_features=max_features,
        max_evaluations=max_evaluations, max_dense_bytes=max_dense_bytes,
        class_index=class_index, seed=seed)
