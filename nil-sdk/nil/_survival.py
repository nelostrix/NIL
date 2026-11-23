import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from nil.runtime import require


class CoxModel(BaseEstimator):
    _nil_backend_distributions = ('lifelines', 'pandas')
    def __init__(self, *, penalizer=.1): self.penalizer = penalizer

    def _frame(self, X, fitted=False):
        values = check_array(X, dtype=float)
        if fitted and values.shape[1] != self.n_features_in_: raise ValueError('Feature count differs')
        return require('pandas', 'survival').DataFrame(values, columns=[f'x{i}' for i in range(values.shape[1])])

    def fit(self, X, y):
        from nil.survival import _outcomes
        if isinstance(self.penalizer, bool) or not np.isfinite(self.penalizer) or self.penalizer < 0: raise ValueError('penalizer must be finite and nonnegative')
        durations, events = _outcomes(*y)
        frame = self._frame(X)
        if len(frame) != len(durations) or not events.any(): raise ValueError('Need matching rows and at least one event')
        frame['__time'], frame['__event'] = durations, events
        fitter = require('lifelines', 'survival').CoxPHFitter(penalizer=self.penalizer)
        fitter.fit(frame, duration_col='__time', event_col='__event')
        self.fitter_, self.n_features_in_ = fitter, frame.shape[1]-2
        return self

    def predict(self, X):
        check_is_fitted(self, 'fitter_')
        return self.fitter_.predict_partial_hazard(self._frame(X, True)).to_numpy()

    def survival_function(self, X, times):
        check_is_fitted(self, 'fitter_')
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (times < 0).any() or (np.diff(times) < 0).any(): raise ValueError('times must be sorted finite nonnegative values')
        return self.fitter_.predict_survival_function(self._frame(X, True), times=times).to_numpy().T


class KaplanMeier(BaseEstimator):
    _nil_backend_distributions = ('lifelines', 'pandas')
    def fit(self, X, y=None):
        from nil.survival import _outcomes
        durations, events = _outcomes(X, y)
        self.fitter_ = require('lifelines', 'survival').KaplanMeierFitter().fit(durations, events)
        return self

    def predict(self, times):
        check_is_fitted(self, 'fitter_')
        values = np.asarray(times, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any(): raise ValueError('times must be nonnegative and finite')
        return self.fitter_.predict(values).to_numpy()
