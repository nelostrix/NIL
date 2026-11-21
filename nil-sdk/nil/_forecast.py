import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted
from nil._classical_common import positive_int


class LagForecaster(BaseEstimator):
    def __init__(self, *, lags=12, horizon=1, estimator=None):
        self.lags, self.horizon, self.estimator = lags, horizon, estimator

    def fit(self, series, y=None, *, dates=None):
        from datetime import date
        from nil.forecasting import date_features
        positive_int(self.lags, 'lags'); positive_int(self.horizon, 'horizon')
        values = np.asarray(series, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all() or len(values) < self.lags+self.horizon:
            raise ValueError('Need a finite univariate series of at least lags+horizon rows')
        origins = range(self.lags, len(values)-self.horizon+1)
        windows = np.lib.stride_tricks.sliding_window_view(values, self.lags+self.horizon)
        x, targets = windows[:, :self.lags], windows[:, self.lags:]
        last_date, date_step = None, None
        if dates is not None:
            if len(dates) != len(values): raise ValueError('dates must match series length')
            parsed = [date.fromisoformat(str(item)) for item in dates]
            steps = [(b-a).days for a,b in zip(parsed, parsed[1:])]
            if not steps or min(steps) <= 0 or len(set(steps)) != 1:
                raise ValueError('dates must increase at a fixed positive day cadence')
            x = np.column_stack([x, date_features([dates[i] for i in origins])])
            last_date, date_step = parsed[-1], steps[0]
        estimator = clone(self.estimator) if self.estimator is not None else Ridge()
        estimator.fit(x, targets if self.horizon > 1 else targets[:, 0])
        self.estimator_, self.history_ = estimator, values[-self.lags:].copy()
        self.last_date_, self.date_step_ = last_date, date_step
        self.n_features_in_, self.training_count_ = x.shape[1], len(values)
        return self

    def predict(self, steps=None):
        from datetime import timedelta
        from nil.forecasting import date_features
        check_is_fitted(self, 'estimator_')
        steps = self.horizon if steps is None else steps
        positive_int(steps, 'steps')
        if steps > self.horizon: raise ValueError('steps cannot exceed the fitted direct horizon')
        x = self.history_[None, :]
        if self.last_date_ is not None:
            x = np.column_stack([x, date_features([(self.last_date_+timedelta(days=self.date_step_)).isoformat()])])
        predicted = np.asarray(self.estimator_.predict(x), dtype=float).reshape(-1)
        if predicted.shape != (self.horizon,) or not np.isfinite(predicted).all():
            raise ValueError('Estimator returned invalid horizon predictions')
        return predicted[:steps]
