"""Optional bounded, resumable Bayesian search with fold-isolated preprocessing."""

import math

from nil._classical_common import positive_int, unwrap
from nil.runtime import require


def _space(space):
    if not isinstance(space, dict) or not space:
        raise ValueError("Provide a nonempty parameter-space mapping")
    for name, spec in space.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Parameter names must be nonempty strings")
        if isinstance(spec, list):
            if not spec or any(type(item) not in {str, int, float, bool, type(None)} or isinstance(item, float) and not math.isfinite(item) for item in spec):
                raise ValueError("Categorical choices must be nonempty JSON scalar lists")
        elif isinstance(spec, dict):
            if spec.get("type") not in {"int", "float"} or set(spec) - {"type", "low", "high", "log", "step"}:
                raise ValueError("Numeric spaces use type, low, high, log and optional step")
            low, high = spec.get("low"), spec.get("high")
            if type(low) not in {int, float} or type(high) not in {int, float} or not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError("Numeric bounds must be finite and ordered")
            if type(spec.get("log", False)) is not bool:
                raise ValueError("log must be a boolean")
            if spec.get("log") and (low <= 0 or "step" in spec):
                raise ValueError("Log spaces need positive bounds and cannot specify step")
            if "step" in spec and (type(spec["step"]) not in {int, float} or not math.isfinite(spec["step"]) or spec["step"] <= 0):
                raise ValueError("step must be positive and finite")
            if spec["type"] == "int" and any(type(spec[key]) is not int for key in ("low", "high", *(["step"] if "step" in spec else []))):
                raise ValueError("Integer spaces need integer bounds and step")
        else:
            raise ValueError("Each parameter needs categorical choices or a numeric specification")


def bayesian(estimator, features, targets, space, *, max_trials=10, cv=3,
             scoring=None, groups=None, sample_weight=None, seed=0, split_method=None,
             gap=0, max_fits=100, output_dir="./nil_runs", storage=None, study_name="nil-search"):
    """TPE search; max_trials is a total persisted attempt budget, including failures.

    One caller owns a study's trial budget. Database persistence supports restart;
    this API does not promise a global quota across simultaneous worker processes.
    """
    from nil.validation import _plan, _scorers, cross_validate, SearchResult, SearchFailed
    estimator = unwrap(estimator)
    positive_int(max_trials, "max_trials")
    positive_int(max_fits, "max_fits")
    _space(space)
    plan = _plan(estimator, features, targets, cv, groups, split_method, seed, gap, max_fits)
    if max_trials * len(plan) + 1 > max_fits:
        raise ValueError("Bayesian search plus final refit exceeds max_fits")
    scorers = _scorers(estimator, scoring, targets)
    if len(scorers) != 1:
        raise ValueError("Tuning requires one explicit selection metric")
    if not isinstance(study_name, str) or not study_name or len(study_name) > 100:
        raise ValueError("study_name must have 1 to 100 characters")
    joblib = require("joblib", "classical")
    optuna = require("optuna", "search")
    # Bind persisted trials to exact inputs, weights, folds, estimator and score.
    # Changing any of these creates a different study rather than mixing results.
    versions = (require("sklearn", "classical").__version__,
                require("numpy", "classical").__version__, optuna.__version__)
    from nil import __version__
    fingerprint = joblib.hash(("nil-tpe-v1", __version__, estimator, features, targets, groups,
                              sample_weight, plan, space, scorers, seed, versions))
    name = study_name + "-" + fingerprint
    study = optuna.create_study(storage=storage, study_name=name, direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=2), load_if_exists=storage is not None)
    base = require("sklearn.base", "classical")
    metric = next(iter(scorers))

    class TrialFailed(Exception):
        pass

    def objective(trial):
        parameters = {}
        for key, spec in space.items():
            if isinstance(spec, list):
                parameters[key] = trial.suggest_categorical(key, spec)
            else:
                options = {key: value for key, value in spec.items() if key != "type"}
                suggest = trial.suggest_int if spec["type"] == "int" else trial.suggest_float
                parameters[key] = suggest(key, **options)
        try:
            model = base.clone(estimator).set_params(**parameters)
            result = cross_validate(model, features, targets, cv=plan, groups=groups,
                sample_weight=sample_weight, scoring=scorers, n_jobs=1, max_fits=max_fits)
            trial.set_user_attr("folds", result.folds)
            return result.summary[metric]["mean"]
        except Exception as error:
            if isinstance(error, (MemoryError, RecursionError)):
                raise
            trial.set_user_attr("error", str(error))
            raise TrialFailed(str(error)) from error

    existing = study.get_trials(deepcopy=False)
    if len(existing) > max_trials:
        raise ValueError("Persisted study already exceeds the requested total max_trials")
    remaining = max_trials - len(existing)
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, catch=(TrialFailed,))
    trials = [{"number": trial.number, "params": trial.params,
               "status": {"COMPLETE": "completed", "FAIL": "failed"}.get(trial.state.name, trial.state.name.lower()),
               "score": trial.value, **trial.user_attrs} for trial in study.get_trials(deepcopy=False)]
    if not any(item["status"] == "completed" for item in trials):
        raise SearchFailed(trials)
    from nil.classical import fit
    fitted = fit(base.clone(estimator).set_params(**study.best_params), features, targets,
                 sample_weight=sample_weight, output_dir=output_dir, name="bayesian-search")
    fitted.run.event("search.completed", {"method": "bayesian", "study_name": name,
                                         "metric": metric, "best_score": study.best_value,
                                         "best_params": study.best_params})
    return SearchResult(fitted, study.best_params, study.best_value, metric, trials, name)


def distributed_search(estimator, features, targets, space, *, storage, max_trials=10,
                       cv=3, scoring=None, groups=None, sample_weight=None, seed=0,
                       split_method=None, gap=0, max_fits=100):
    """A shared SQLite random-search ledger; every worker must supply the same workload.

    Use search.run(...cv_plan=search.cv_plan, scoring=search.scoring) in each
    process. Scores are selected from persisted trials; final refitting is explicit.
    Expired attempts consume their budget and are never silently repeated.
    """
    from nil.validation import _plan,_scorers
    from nil._distributed_search import DistributedSearch
    estimator=unwrap(estimator); positive_int(max_trials,'max_trials'); positive_int(max_fits,'max_fits')
    _space(space)
    plan=_plan(estimator,features,targets,cv,groups,split_method,seed,gap,max_fits)
    if max_trials*len(plan)>max_fits: raise ValueError('Distributed trial budget exceeds max_fits')
    scorers=_scorers(estimator,scoring,targets)
    if len(scorers)!=1: raise ValueError('Distributed tuning requires one selection metric')
    fingerprint=_distributed_fingerprint(estimator,features,targets,plan,groups,sample_weight,scorers)
    result=DistributedSearch(storage,space=space,max_trials=max_trials,fingerprint=fingerprint,seed=seed)
    result.cv_plan,result.scoring=plan,scorers
    return result


def _distributed_fingerprint(estimator,features,targets,plan,groups,sample_weight,scorers):
    from nil import __version__
    import platform
    return require('joblib','classical').hash(('nil-distributed-v1',platform.python_version(),require('joblib','classical').__version__,__version__,estimator,features,targets,plan,groups,sample_weight,
        scorers,require('sklearn','classical').__version__,require('numpy','classical').__version__))
