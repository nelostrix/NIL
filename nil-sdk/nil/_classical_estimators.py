"""Optional sklearn-protocol estimators, imported only when selected."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted


class _OnlineBase(BaseEstimator):
    def __init__(self, estimator, *, preprocess=None, classes=None):
        self.estimator = estimator
        self.preprocess = preprocess
        self.classes = classes

    def __sklearn_is_fitted__(self):
        return hasattr(self, "estimator_") and getattr(self, "batch_count_", 0) > 0 and not getattr(self, "failed_", False)

    def fit(self, X, y=None, sample_weight=None):
        for name in ("estimator_", "preprocess_", "batch_count_", "classes_", "failed_"):
            self.__dict__.pop(name, None)
        return self.partial_fit(X, y, sample_weight=sample_weight)

    def partial_fit(self, X, y=None, sample_weight=None):
        from nil._classical_common import weights_for
        from sklearn.utils.validation import check_consistent_length
        if getattr(self, "failed_", False):
            raise RuntimeError("Previous batch failed; restore a checkpoint or restart with fit before continuing")
        check_consistent_length(X, y)
        weight = weights_for(sample_weight, y if y is not None else X)
        if not hasattr(self.estimator, "partial_fit"):
            raise TypeError("Select an estimator with partial_fit support")
        first = not hasattr(self, "estimator_")
        if first and isinstance(self, ClassifierMixin) and self.classes is None:
            raise ValueError("Provide all possible classes= before the first classification batch")
        try:
            if first:
                self.estimator_ = clone(self.estimator)
                self.preprocess_ = clone(self.preprocess) if self.preprocess is not None else None
                self.batch_count_ = 0
                transformed = self.preprocess_.fit_transform(X, y) if self.preprocess_ is not None else X
            else:
                # Freeze learned coordinates. Updating a scaler underneath an
                # already trained linear model changes the meaning of weights.
                transformed = self._transform(X)
            options = {"sample_weight": weight} if weight is not None else {}
            if first and isinstance(self, ClassifierMixin):
                options["classes"] = self.classes
            self.estimator_.partial_fit(transformed, y, **options)
            if hasattr(self.estimator_, "classes_"):
                self.classes_ = self.estimator_.classes_
            self.batch_count_ += 1
            self.failed_ = False
        except BaseException:
            self.failed_ = True
            raise
        return self

    def _transform(self, X):
        return self.preprocess_.transform(X) if self.preprocess_ is not None else X

    def predict(self, X):
        check_is_fitted(self)
        return self.estimator_.predict(self._transform(X))

    @available_if(lambda self: hasattr(self.estimator, "predict_proba"))
    def predict_proba(self, X):
        check_is_fitted(self)
        return self.estimator_.predict_proba(self._transform(X))

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    def decision_function(self, X):
        check_is_fitted(self)
        return self.estimator_.decision_function(self._transform(X))

    def checkpoint(self, path):
        from nil.persistence import save
        check_is_fitted(self)
        return save(self, path)


class OnlineClassifier(ClassifierMixin, _OnlineBase):
    pass


class OnlineRegressor(RegressorMixin, _OnlineBase):
    pass


class ConformalRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator, *, alpha=0.1):
        self.estimator = estimator
        self.alpha = alpha

    def fit(self, X, y, **parameters):
        self.__dict__.pop("radius_", None)
        self.__dict__.pop("calibration_count_", None)
        self.estimator_ = clone(self.estimator).fit(X, y, **parameters)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def interval(self, X):
        check_is_fitted(self, "radius_")
        prediction = self.predict(X)
        return np.column_stack((prediction - self.radius_, prediction + self.radius_))


class ConformalClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, *, alpha=0.1):
        self.estimator = estimator
        self.alpha = alpha

    def fit(self, X, y, **parameters):
        self.__dict__.pop("cutoff_", None)
        self.__dict__.pop("calibration_count_", None)
        self.estimator_ = clone(self.estimator).fit(X, y, **parameters)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        from nil.uncertainty import _probabilities
        check_is_fitted(self, "estimator_")
        return _probabilities(self.estimator_.predict_proba(X), len(self.classes_))

    def prediction_sets(self, X):
        check_is_fitted(self, "cutoff_")
        return 1 - self.predict_proba(X) <= self.cutoff_


class ThresholdClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, *, threshold=0.5):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y, **parameters):
        self.estimator_ = clone(self.estimator).fit(X, y, **parameters)
        self.classes_ = self.estimator_.classes_
        if len(self.classes_) != 2:
            raise ValueError("Threshold selection requires binary classification")
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= self.threshold).astype(int)]
