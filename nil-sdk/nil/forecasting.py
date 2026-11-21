"""Direct lag/horizon forecasting and chronological rolling-origin evaluation."""
from nil.runtime import require
from nil._classical_common import positive_int


def date_features(dates):
    """Calendar features for ISO dates; fixed day-of-week/month cycles, no fitted state."""
    from datetime import date
    np = require('numpy', 'classical')
    parsed = [date.fromisoformat(str(value)) for value in dates]
    return np.asarray([[d.toordinal(), np.sin(2*np.pi*d.weekday()/7), np.cos(2*np.pi*d.weekday()/7),
                        np.sin(2*np.pi*(d.month-1)/12), np.cos(2*np.pi*(d.month-1)/12)] for d in parsed], dtype=float)


def model(*, lags=12, horizon=1, estimator=None):
    require('sklearn','classical')
    from nil._forecast import LagForecaster
    return LagForecaster(lags=lags, horizon=horizon, estimator=estimator)


def fit(series, *, lags=12, horizon=1, estimator=None, dates=None):
    return model(lags=lags, horizon=horizon, estimator=estimator).fit(series, dates=dates)


def backtest(series, *, initial_train, horizon=1, step=1, lags=12, gap=0,
             estimator=None, dates=None, max_fits=100):
    np = require('numpy', 'classical')
    for name, value in [('initial_train', initial_train), ('horizon', horizon), ('step', step), ('lags', lags), ('max_fits', max_fits)]:
        positive_int(value, name)
    if type(gap) is not int or gap < 0:
        raise ValueError('gap must be a nonnegative integer')
    values = np.asarray(series, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('series must be a finite univariate sequence')
    if dates is not None and len(dates) != len(values):
        raise ValueError('dates must match series length')
    if dates is not None:
        ordinal=date_features(dates)[:,0]
        differences=np.diff(ordinal)
        if not len(differences) or (differences<=0).any() or not np.all(differences==differences[0]):
            raise ValueError('Backtest dates require a fixed positive day cadence')
    origins = range(initial_train + gap, len(values)-horizon+1, step)
    if not origins or len(origins) > max_fits:
        raise ValueError('No backtest folds or max_fits exceeded')
    folds = []
    for origin in origins:
        end = origin-gap
        trained = fit(values[:end], lags=lags, horizon=horizon+gap, estimator=estimator,
                      dates=dates[:end] if dates is not None else None)
        predicted = trained.predict()[gap:]
        actual = values[origin:origin+horizon]
        folds.append({'train_end': end, 'test_start': origin, 'test_end': origin+horizon,
                      'prediction': predicted.tolist(), 'actual': actual.tolist()})
    errors = np.asarray([np.asarray(row['prediction'])-row['actual'] for row in folds])
    return {'folds': folds, 'mae': float(np.abs(errors).mean()), 'rmse': float(np.sqrt((errors**2).mean()))}
