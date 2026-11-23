"""Classical fitting/results and held-out metrics; native engines remain lazy."""
from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import warnings

from nil._classical_common import is_classical, task_for, unwrap, warning_records, weighted_fit_params, weights_for
from nil.lifecycle import Run
from nil.runtime import require


@dataclass
class ClassicalResult(Mapping):
    """Attribute access plus compatibility with result['model'/'run'/'metrics']."""
    model: object
    run: Run
    metrics: dict
    task: str
    warnings: list = field(default_factory=list)
    status: str = "completed"
    _nil_classical_result = True

    def __getitem__(self, key):
        if key not in ("model", "run", "metrics", "task", "warnings", "status"):
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(("model", "run", "metrics", "task", "warnings", "status"))

    def __len__(self):
        return 6

    def predict(self, features):
        return self.model.predict(features)

    def predict_proba(self, features):
        return self.model.predict_proba(features)

    def transform(self, features):
        return self.model.transform(features)

    def evaluate(self, features, targets=None, **options):
        return evaluate(self.model, features, targets, task=self.task, **options)

    def save(self, path):
        from nil.persistence import save
        return save(self, path)


def evaluate(model, features, targets=None, *, task=None, sample_weight=None, cluster_labels=None, seed=0):
    """Evaluate without fitting, changing model state or refitting preprocessing."""
    model = unwrap(model)
    task = task_for(model, task)
    metrics = require("sklearn.metrics", "classical")
    np = require("numpy", "classical")
    if task == "transformation":
        raise ValueError("Transformers have no default predictive metrics; evaluate a downstream pipeline or use an explicit CV scorer")
    if task == "clustering":
        if sample_weight is not None:
            raise ValueError("Clustering evaluation metrics do not support sample_weight")
        labels = model.predict(features) if cluster_labels is None and hasattr(model, "predict") else cluster_labels
        if labels is None:
            raise ValueError("Supply cluster_labels explicitly for transductive clustering models")
        labels = np.asarray(labels)
        utils = require("sklearn.utils", "classical")
        require("sklearn.utils.validation", "classical").check_consistent_length(features, labels)
        if labels.ndim != 1 or not len(labels):
            raise ValueError("Cluster labels must be nonempty and one-dimensional")
        result = {"count": len(labels), "clusters": int(len(set(labels) - {-1})), "noise_count": int((labels == -1).sum())}
        valid = labels != -1
        unique = np.unique(labels[valid])
        if 1 < len(unique) < int(valid.sum()):
            metric_features = model[:-1].transform(features) if hasattr(model, "steps") and len(model.steps) > 1 else features
            chosen = np.flatnonzero(valid)
            if len(chosen) > 2000:
                chosen = np.random.RandomState(seed).choice(chosen, size=2000, replace=False)
            sample_labels = labels[chosen]
            result["silhouette_sample_count"] = len(chosen)
            result["silhouette"] = (float(metrics.silhouette_score(utils._safe_indexing(metric_features, chosen), sample_labels))
                                     if 1 < len(np.unique(sample_labels)) < len(chosen) else None)
        if targets is not None:
            require("sklearn.utils.validation", "classical").check_consistent_length(targets, labels)
            result["adjusted_rand"] = float(metrics.adjusted_rand_score(targets, labels))
            result["normalized_mutual_info"] = float(metrics.normalized_mutual_info_score(targets, labels))
        return result
    predicted = model.predict(features)
    if task == "anomaly_detection" and targets is None:
        weight = weights_for(sample_weight, predicted)
        return {"count": len(predicted), "outlier_fraction": float(np.average(np.asarray(predicted) == -1, weights=weight))}
    if targets is None:
        raise ValueError("Held-out targets are required for supervised evaluation")
    weight = weights_for(sample_weight, targets)
    if not len(targets):
        raise ValueError("Evaluation data must not be empty")
    options = {"sample_weight": weight}
    if task == "regression":
        result = {
            "count": len(targets),
            "mae": float(metrics.mean_absolute_error(targets, predicted, **options)),
            "mse": float(metrics.mean_squared_error(targets, predicted, **options)),
            "rmse": float(metrics.root_mean_squared_error(targets, predicted, **options)),
            "r2": float(metrics.r2_score(targets, predicted, **options)) if len(targets) > 1 else None,
        }
    else:
        if np.asarray(targets).ndim != 1:
            return _multioutput_metrics(model, features, targets, predicted, weight)
        labels = getattr(model, "classes_", None)
        if labels is None:
            labels = np.unique(np.concatenate([np.asarray(targets), np.asarray(predicted)]))
        elif not np.isin(np.asarray(targets), labels).all():
            raise ValueError("Held-out targets contain classes absent from training; revise the split")
        precision, recall, f1, support = metrics.precision_recall_fscore_support(targets, predicted, labels=labels, zero_division=0, **options)
        result = {
            "count": len(targets), "accuracy": float(metrics.accuracy_score(targets, predicted, **options)),
            "labels": np.asarray(labels).tolist(),
            "confusion_matrix": metrics.confusion_matrix(targets, predicted, labels=labels, **options).tolist(),
            "per_class": {"precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(), "support": support.tolist()},
        }
        for average in ("macro", "weighted"):
            p, r, f, _ = metrics.precision_recall_fscore_support(targets, predicted, average=average, zero_division=0, **options)
            result.update({f"precision_{average}": float(p), f"recall_{average}": float(r), f"f1_{average}": float(f)})
        unique = np.unique(targets)
        result["balanced_accuracy"] = float(metrics.balanced_accuracy_score(targets, predicted, **options)) if len(unique) > 1 else None
        result["matthews_correlation"] = float(metrics.matthews_corrcoef(targets, predicted, **options))
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            result["log_loss"] = float(metrics.log_loss(targets, probabilities, labels=labels, **options))
            if len(labels) == 2 and len(unique) == 2:
                binary = np.asarray(targets) == labels[1]
                result["roc_auc"] = float(metrics.roc_auc_score(binary, probabilities[:, 1], **options))
                result["average_precision"] = float(metrics.average_precision_score(binary, probabilities[:, 1], **options))
                result["brier"] = float(metrics.brier_score_loss(binary, probabilities[:, 1], **options))
            elif len(labels) > 2 and len(unique) == len(labels):
                result["roc_auc_ovr_weighted"] = float(metrics.roc_auc_score(targets, probabilities, labels=labels, multi_class="ovr", average="weighted", **options))
        elif hasattr(model, "decision_function") and len(labels) == 2 and len(unique) == 2:
            result["roc_auc"] = float(metrics.roc_auc_score(np.asarray(targets) == labels[1], model.decision_function(features), **options))
    if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
        raise ValueError("Evaluation produced a non-finite metric")
    return result


def _multioutput_metrics(model, features, targets, predicted, weight):
    np = require("numpy", "classical")
    metrics = require("sklearn.metrics", "classical")
    actual, predicted = np.asarray(targets), np.asarray(predicted)
    if actual.ndim != 2 or actual.shape != predicted.shape or not actual.shape[1]:
        raise ValueError("Multioutput targets and predictions must be matching nonempty matrices")
    options = {"sample_weight": weight}
    result = {"count": len(actual), "output_count": actual.shape[1],
              "subset_accuracy": float(np.average(np.all(actual == predicted, axis=1), weights=weight)),
              "hamming_loss": float(np.average(np.mean(actual != predicted, axis=1), weights=weight)),
              "per_output": []}
    classes = getattr(model, "classes_", None)
    probabilities = model.predict_proba(features) if hasattr(model, "predict_proba") else None
    for index in range(actual.shape[1]):
        labels = classes[index] if isinstance(classes, list) else np.unique(np.concatenate((actual[:, index], predicted[:, index])))
        if not np.isin(actual[:, index], labels).all():
            raise ValueError("Held-out targets contain classes absent from training")
        p, r, f, _ = metrics.precision_recall_fscore_support(actual[:, index], predicted[:, index],
                                                         average="macro", zero_division=0, **options)
        report = {"accuracy": float(metrics.accuracy_score(actual[:, index], predicted[:, index], **options)),
                  "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
                  "labels": np.asarray(labels).tolist(),
                  "confusion_matrix": metrics.confusion_matrix(actual[:, index], predicted[:, index], labels=labels, **options).tolist()}
        if isinstance(probabilities, list):
            report["log_loss"] = float(metrics.log_loss(actual[:, index], probabilities[index], labels=labels, **options)) if len(labels) > 1 else None
        result["per_output"].append(report)
    kind = require("sklearn.utils.multiclass", "classical").type_of_target(actual)
    if kind == "multilabel-indicator":
        for average in ("micro", "macro", "weighted", "samples"):
            p, r, f, _ = metrics.precision_recall_fscore_support(actual, predicted, average=average, zero_division=0, **options)
            result.update({f"precision_{average}": float(p), f"recall_{average}": float(r), f"f1_{average}": float(f)})
    return result


def fit(estimator, features, targets=None, *, validation=None, task=None,
        sample_weight=None, validation_weight=None, output_dir="./nil_runs", name="classical", **fit_params):
    estimator = unwrap(estimator)
    if not is_classical(estimator):
        raise TypeError("nil.fit handles native classical estimators; use nil.train for neural modules or nil.rl.train for control algorithms")
    task = task_for(estimator, task)
    if validation_weight is not None and validation is None:
        raise ValueError("validation_weight requires held-out validation data")
    if validation is not None and (not isinstance(validation, (tuple, list)) or len(validation) != 2):
        raise ValueError("validation must be a (features, targets) pair")
    weight = weights_for(sample_weight, targets if targets is not None else features)
    params = weighted_fit_params(estimator, weight, fit_params)
    run = Run(output_dir, name, {"estimator": type(estimator).__name__, "task": task})
    run.status("running")
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            estimator.fit(features, targets, **params)
        notices = warning_records(recorded)
        scores = evaluate(estimator, *validation, task=task, sample_weight=validation_weight) if validation is not None else {}
        if scores:
            run.log({key: value for key, value in scores.items() if isinstance(value, (int, float))}, 1)
            run.event("evaluation", scores)
        for notice in notices:
            run.event("warning", notice)
        run.status("completed")
        return ClassicalResult(estimator, run, scores, task, notices)
    except BaseException as exc:
        run.status("cancelled" if isinstance(exc, KeyboardInterrupt) else "failed", error=type(exc).__name__, message=str(exc))
        raise
