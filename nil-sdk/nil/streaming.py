"""Incremental training with frozen preprocessing and complete checkpoints."""

from nil._classical_common import unwrap
from nil.runtime import require


def create(estimator, *, preprocess=None, classes=None):
    estimator = unwrap(estimator)
    base = require("sklearn.base", "classical")
    if not callable(getattr(estimator, "partial_fit", None)):
        raise TypeError("Select an estimator with partial_fit support, for example sgd_classifier")
    from nil._classical_estimators import OnlineClassifier, OnlineRegressor
    if base.is_classifier(estimator):
        if classes is None:
            raise ValueError("Provide all possible classes= for incremental classification")
        return OnlineClassifier(estimator, preprocess=preprocess, classes=classes)
    if base.is_regressor(estimator):
        return OnlineRegressor(estimator, preprocess=preprocess)
    raise ValueError("Streaming currently supports supervised incremental classifiers and regressors")


def resume(path, *, trusted=False, allow_version_mismatch=False):
    from nil.persistence import load
    model = load(path, trusted=trusted, allow_version_mismatch=allow_version_mismatch)
    from nil._classical_estimators import OnlineClassifier, OnlineRegressor
    if not isinstance(model, (OnlineClassifier, OnlineRegressor)):
        raise ValueError("Artifact is not a NIL streaming checkpoint")
    return model
