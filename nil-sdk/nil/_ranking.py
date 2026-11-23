import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import check_is_fitted, check_consistent_length
from nil.runtime import require
from nil._classical_common import weights_for


class QueryRanker(BaseEstimator):
    @property
    def _nil_backend_distributions(self):
        return tuple(name for name in ('lightgbm','xgboost','catboost') if type(getattr(self,'estimator_',None)).__module__.startswith(name))
    def __init__(self, estimator=None): self.estimator = estimator

    def fit(self, X, y, *, groups, sample_weight=None):
        from nil.ranking import _queries
        check_consistent_length(X, y, groups)
        actual = np.asarray(y, dtype=float)
        if actual.ndim != 1 or not np.isfinite(actual).all() or (actual < 0).any() or (actual > 31).any() or (actual != actual.astype(int)).any():
            raise ValueError('Native ranker relevance must be integer grades in [0,31]')
        queries = _queries(groups, len(actual)); order = np.concatenate(queries)
        weights = weights_for(sample_weight, y)
        estimator = clone(self.estimator) if self.estimator is not None else require('lightgbm', 'lightgbm').LGBMRanker(n_jobs=1, verbosity=-1)
        estimator.fit(_safe_indexing(X, order), actual[order], group=[len(q) for q in queries],
                      **({'sample_weight': weights[order]} if weights is not None else {}))
        self.estimator_, self.n_features_in_ = estimator, estimator.n_features_in_
        return self

    def predict(self, X):
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict(X)
