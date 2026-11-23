"""Reproducible splits, fold-local evaluation and explicitly bounded tuning."""
from dataclasses import dataclass
import itertools
import math
from numbers import Integral, Real
import time
import warnings

from nil._classical_common import positive_int, task_for, unwrap, warning_records, weighted_fit_params, weights_for
from nil.runtime import require


@dataclass
class DataSplit:
    X_train: object
    X_test: object
    y_train: object
    y_test: object
    train_indices: object
    test_indices: object

    @property
    def train(self):
        return self.X_train, self.y_train

    @property
    def test(self):
        return self.X_test, self.y_test


def split(features, targets=None, *, method="random", test_size=0.2, seed=0, groups=None, gap=0):
    selection = require("sklearn.model_selection", "classical")
    utils = require("sklearn.utils", "classical")
    np = require("numpy", "classical")
    require("sklearn.utils.validation", "classical").check_consistent_length(features, targets, groups)
    n = features.shape[0] if hasattr(features, "shape") else len(features)
    indices = np.arange(n)
    if method in {"random", "stratified"}:
        if groups is not None:
            raise ValueError("Use method='group' when supplying groups")
        if gap:
            raise ValueError("gap applies only to time splits")
        if method == "stratified" and targets is None:
            raise ValueError("Stratification requires targets")
        train, test = selection.train_test_split(indices, test_size=test_size, random_state=seed,
                                                 stratify=targets if method == "stratified" else None)
    elif method == "group":
        if groups is None:
            raise ValueError("Group splitting requires groups")
        if gap:
            raise ValueError("gap applies only to time splits")
        train, test = next(selection.GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(features, targets, groups))
    elif method == "time":
        if isinstance(gap, bool) or not isinstance(gap, Integral) or gap < 0:
            raise ValueError("gap must be a nonnegative integer")
        if isinstance(test_size, Real) and not isinstance(test_size, Integral):
            if not 0 < test_size < 1:
                raise ValueError("Fractional test_size must be between zero and one")
            count = math.ceil(n * test_size)
        else:
            count = positive_int(test_size, "test_size")
        end = n - count - gap
        if end < 1:
            raise ValueError("Time split leaves no training samples")
        train, test = indices[:end], indices[n - count:]
        if groups is not None and np.intersect1d(np.asarray(groups)[train], np.asarray(groups)[test]).size:
            raise ValueError("Time split crosses supplied groups")
    else:
        raise ValueError("Choose random, stratified, group or time")
    take = utils._safe_indexing
    return DataSplit(take(features, train), take(features, test),
                     take(targets, train) if targets is not None else None,
                     take(targets, test) if targets is not None else None, train, test)


def folds(*, method="stratified", n_splits=5, seed=0, gap=0):
    positive_int(n_splits, "n_splits")
    selection = require("sklearn.model_selection", "classical")
    if method != "time" and gap:
        raise ValueError("gap applies only to time folds")
    if method == "random":
        return selection.KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if method == "stratified":
        return selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if method == "group":
        return selection.GroupKFold(n_splits=n_splits)
    if method == "stratified_group":
        return selection.StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if method == "time":
        return selection.TimeSeriesSplit(n_splits=n_splits, gap=gap)
    raise ValueError("Choose random, stratified, group, stratified_group or time")


def _plan(estimator, features, targets, cv, groups, method, seed, gap, max_fits):
    positive_int(max_fits, "max_fits")
    utils = require("sklearn.utils", "classical")
    np = require("numpy", "classical")
    require("sklearn.utils.validation", "classical").check_consistent_length(features, targets, groups)
    n = features.shape[0] if hasattr(features, "shape") else len(features)
    if isinstance(cv, Integral) and not isinstance(cv, bool):
        if method is None:
            classification = require("sklearn.base", "classical").is_classifier(estimator)
            if classification and targets is not None:
                classification = require("sklearn.utils.multiclass", "classical").type_of_target(targets) in {"binary", "multiclass"}
            method = ("stratified_group" if classification else "group") if groups is not None else ("stratified" if classification else "random")
        cv = folds(method=method, n_splits=cv, seed=seed, gap=gap)
    elif method is not None or gap:
        raise ValueError("method/gap configure integer cv; pass a configured splitter or fold iterable alone")
    source = cv.split(features, targets, groups) if hasattr(cv, "split") else iter(cv)
    plan = list(itertools.islice(source, max_fits + 1))
    if not plan or len(plan) > max_fits:
        raise ValueError("Cross-validation exceeds max_fits or has no folds")
    normalized = []
    for train, test in plan:
        train, test = np.asarray(train), np.asarray(test)
        for part in (train, test):
            if part.ndim != 1 or not len(part) or not np.issubdtype(part.dtype, np.integer):
                raise ValueError("Fold indices must be nonempty integer arrays")
            if part.min() < 0 or part.max() >= n or len(np.unique(part)) != len(part):
                raise ValueError("Fold indices are out of bounds or duplicated")
        if np.intersect1d(train, test).size:
            raise ValueError("Training and test folds overlap")
        if groups is not None and np.intersect1d(utils._safe_indexing(groups, train), utils._safe_indexing(groups, test)).size:
            raise ValueError("A group appears in both training and test folds")
        normalized.append((train, test))
    return normalized


def _subset_accuracy(model, features, targets, sample_weight=None):
    np = require("numpy", "classical")
    return np.average(np.all(np.asarray(targets) == model.predict(features), axis=1), weights=sample_weight)


def _scorers(estimator, scoring, targets=None):
    metrics = require("sklearn.metrics", "classical")
    if scoring is None:
        task = task_for(estimator)
        if task not in {"classification", "regression"}:
            raise ValueError("Supply scoring explicitly for unsupervised model selection")
        if task == "classification" and targets is not None and require("sklearn.utils.multiclass", "classical").type_of_target(targets) == "multiclass-multioutput":
            return {"subset_accuracy": _subset_accuracy}
        scoring = "accuracy" if task == "classification" else "neg_root_mean_squared_error"
    if isinstance(scoring, str):
        return {scoring: metrics.get_scorer(scoring)}
    if callable(scoring):
        return {"score": scoring}
    if isinstance(scoring, (list, tuple)) and scoring:
        return {name: metrics.get_scorer(name) for name in scoring}
    if isinstance(scoring, dict) and scoring:
        return {name: metrics.get_scorer(value) if isinstance(value, str) else value for name, value in scoring.items()}
    raise ValueError("scoring must be a native scorer name, names, or a nonempty scorer mapping")


@dataclass
class CVResult:
    scores: dict
    summary: dict
    folds: list
    train_scores: dict | None = None


def cross_validate(estimator, features, targets=None, *, cv=5, method=None, groups=None,
                   sample_weight=None, scoring=None, seed=0, gap=0, n_jobs=1, max_fits=100,
                   return_train_score=False):
    estimator = unwrap(estimator)
    if not isinstance(n_jobs, Integral) or isinstance(n_jobs, bool) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer; the lightweight default is 1")
    plan = _plan(estimator, features, targets, cv, groups, method, seed, gap, max_fits)
    weight = weights_for(sample_weight, targets if targets is not None else features)
    scorers = _scorers(estimator, scoring, targets)
    base = require("sklearn.base", "classical")
    take = require("sklearn.utils", "classical")._safe_indexing
    np = require("numpy", "classical")

    def run_fold(train, test):
        model = base.clone(estimator)
        X_train, X_test = take(features, train), take(features, test)
        y_train = take(targets, train) if targets is not None else None
        y_test = take(targets, test) if targets is not None else None
        train_weight = take(weight, train) if weight is not None else None
        test_weight = take(weight, test) if weight is not None else None
        params = weighted_fit_params(model, train_weight)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            model.fit(X_train, y_train, **params)
        fit_time = time.perf_counter() - started
        started = time.perf_counter()
        scores, train_scores = {}, {}
        for name, scorer in scorers.items():
            options = {"sample_weight": test_weight} if weight is not None else {}
            value = float(scorer(model, X_test, y_test, **options))
            if not math.isfinite(value):
                raise ValueError(f"Scorer {name!r} produced a non-finite result")
            scores[name] = value
            if return_train_score:
                train_options = {"sample_weight": train_weight} if weight is not None else {}
                train_value = float(scorer(model, X_train, y_train, **train_options))
                if not math.isfinite(train_value):
                    raise ValueError(f"Training scorer {name!r} produced a non-finite result")
                train_scores[name] = train_value
        return {"scores": scores, "train_count": len(train), "test_count": len(test),
                "fit_time": fit_time, "score_time": time.perf_counter() - started,
                "warnings": warning_records(recorded), "train_scores": train_scores}

    if n_jobs == 1:
        results = [run_fold(train, test) for train, test in plan]
    else:
        joblib = require("joblib", "classical")
        results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(run_fold)(train, test) for train, test in plan)
    scores = {name: [item["scores"][name] for item in results] for name in scorers}
    summary = {name: {"mean": float(np.mean(values)), "std": float(np.std(values))} for name, values in scores.items()}
    training = {name: [item["train_scores"][name] for item in results] for name in scorers} if return_train_score else None
    return CVResult(scores, summary, results, training)


@dataclass
class SearchResult:
    result: object
    best_params: dict
    best_score: float
    metric: str
    trials: list
    study_name: str | None = None
    _nil_classical_result = True

    @property
    def model(self):
        return self.result.model

    @property
    def run(self):
        return self.result.run

    @property
    def metrics(self):
        return self.result.metrics

    @property
    def status(self):
        return self.result.status

    @property
    def warnings(self):
        return self.result.warnings

    def predict(self, features):
        return self.model.predict(features)

    def predict_proba(self, features):
        return self.model.predict_proba(features)

    def transform(self, features):
        return self.model.transform(features)

    def evaluate(self, features, targets, **options):
        return self.result.evaluate(features, targets, **options)

    def save(self, path):
        return self.result.save(path)


class SearchFailed(ValueError):
    def __init__(self, trials):
        super().__init__("All search trials failed; inspect exception.trials")
        self.trials = trials


def tune(estimator, features, targets, space, *, method="random", max_trials=10, cv=3,
         scoring=None, groups=None, sample_weight=None, seed=0, split_method=None, gap=0,
         n_jobs=1, max_fits=100, output_dir="./nil_runs", storage=None, study_name="nil-search"):
    """Maximize one native scorer. Loss scorers retain sklearn's negative-score convention."""
    estimator = unwrap(estimator)
    if method == "bayesian":
        if n_jobs != 1:
            raise ValueError("Bayesian search uses one worker per invocation")
        from nil.selection import bayesian
        return bayesian(estimator, features, targets, space, max_trials=max_trials, cv=cv,
            scoring=scoring, groups=groups, sample_weight=sample_weight, seed=seed,
            split_method=split_method, gap=gap, max_fits=max_fits, output_dir=output_dir,
            storage=storage, study_name=study_name)
    if storage is not None or study_name != "nil-search":
        raise ValueError("storage/study_name apply only to Bayesian search")
    positive_int(max_trials, "max_trials")
    positive_int(max_fits, "max_fits")
    selection = require("sklearn.model_selection", "classical")
    base = require("sklearn.base", "classical")
    plan = _plan(estimator, features, targets, cv, groups, split_method, seed, gap, max_fits)
    if method == "grid":
        candidates = selection.ParameterGrid(space)
        if len(candidates) > max_trials:
            raise ValueError("Grid exceeds max_trials; narrow it or raise the explicit budget")
    elif method == "random":
        candidates = selection.ParameterSampler(space, n_iter=max_trials, random_state=seed)
    else:
        raise ValueError("Choose grid, random or bayesian search")
    if len(candidates) * len(plan) + 1 > max_fits:
        raise ValueError("Search plus final refit exceeds max_fits")
    candidates = list(candidates)
    scorers = _scorers(estimator, scoring, targets)
    if len(scorers) != 1:
        raise ValueError("Tuning requires one explicit selection metric")
    metric = next(iter(scorers))
    trials = []
    best = None
    for parameters in candidates:
        try:
            model = base.clone(estimator).set_params(**parameters)
            result = cross_validate(model, features, targets, cv=plan, groups=groups,
                sample_weight=sample_weight, scoring=scorers, seed=seed, n_jobs=n_jobs, max_fits=max_fits)
            score = result.summary[metric]["mean"]
            trials.append({"params": parameters, "status": "completed", "score": score,
                           "folds": result.folds})
            if best is None or score > best[0]:
                best = score, parameters
        except Exception as error:
            if isinstance(error, (MemoryError, RecursionError)):
                raise
            trials.append({"params": parameters, "status": "failed", "error": str(error)})
    if best is None:
        raise SearchFailed(trials)
    from nil.classical import fit
    fitted = fit(base.clone(estimator).set_params(**best[1]), features, targets,
                 sample_weight=sample_weight, output_dir=output_dir, name="classical-search")
    fitted.run.event("search.completed", {"method": method, "metric": metric, "best_score": best[0],
                                         "best_params": {key: repr(value) for key, value in best[1].items()},
                                         "trials": len(trials)})
    return SearchResult(fitted, best[1], best[0], metric, trials)


def nested_validate(estimator, features, targets, space, *, outer_cv=5, inner_cv=3,
                    method="random", split_method=None, groups=None, sample_weight=None,
                    scoring=None, max_trials=10, max_fits=200, seed=0, gap=0,
                    output_dir="./nil_runs"):
    """Tune on each outer training fold; score only its untouched outer test rows."""
    estimator = unwrap(estimator)
    positive_int(inner_cv, "inner_cv")
    positive_int(max_trials, "max_trials")
    if method not in {"grid", "random", "bayesian"}:
        raise ValueError("Choose grid, random or bayesian search")
    plan = _plan(estimator, features, targets, outer_cv, groups, split_method, seed, gap, max_fits)
    per_fold = max_trials * inner_cv + 1
    if len(plan) * per_fold > max_fits:
        raise ValueError("Nested tuning plus refits exceeds max_fits")
    scorers = _scorers(estimator, scoring, targets)
    if len(scorers) != 1:
        raise ValueError("Nested selection requires one metric")
    metric, scorer = next(iter(scorers.items()))
    take = require("sklearn.utils", "classical")._safe_indexing
    np = require("numpy", "classical")
    weight = weights_for(sample_weight, targets)
    results = []
    for index, (train, test) in enumerate(plan):
        trained = tune(estimator, take(features, train), take(targets, train), space,
            method=method, cv=inner_cv, groups=take(groups, train) if groups is not None else None,
            sample_weight=take(weight, train) if weight is not None else None,
            scoring=scorers, split_method=split_method, gap=gap, seed=seed + index,
            max_trials=max_trials, max_fits=per_fold, output_dir=output_dir)
        options = {"sample_weight": take(weight, test)} if weight is not None else {}
        score = float(scorer(trained.model, take(features, test), take(targets, test), **options))
        if not math.isfinite(score):
            raise ValueError("Outer test scorer produced a non-finite result")
        results.append({"score": score, "best_params": trained.best_params,
                        "inner_score": trained.best_score, "train_count": len(train), "test_count": len(test)})
    scores = [record["score"] for record in results]
    return {"metric": metric, "folds": results, "mean": float(np.mean(scores)),
            "std": float(np.std(scores)), "fit_budget": len(plan) * per_fold}
