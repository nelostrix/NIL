"""Use native scikit-learn estimators with NIL runs and evaluation."""

from nil.lifecycle import Run
from nil.runtime import require


def fit(
    estimator,
    features,
    targets,
    *,
    validation=None,
    task="classification",
    output_dir="./nil_runs",
    name="classical",
):
    metrics = require("sklearn.metrics", "classical")
    if task not in {"classification", "regression"}:
        raise ValueError("Choose classification or regression")
    run = Run(output_dir, name, {"estimator": type(estimator).__name__, "task": task})
    run.status("running")
    try:
        estimator.fit(features, targets)
        result = {}
        if validation is not None:
            x, y = validation
            predicted = estimator.predict(x)
            if task == "classification":
                result["accuracy"] = metrics.accuracy_score(y, predicted)
            else:
                result["mse"] = metrics.mean_squared_error(y, predicted)
            run.log(result, 1)
        run.status("completed")
        return {"model": estimator, "run": run, "metrics": result}
    except BaseException as exc:
        run.status("failed", error=type(exc).__name__, message=str(exc))
        raise
