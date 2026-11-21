"""Right-censored survival analysis via an explicit optional lifelines backend."""
from nil.runtime import require


def _outcomes(durations, events):
    np = require('numpy', 'classical')
    t, e = np.asarray(durations, dtype=float), np.asarray(events)
    if t.ndim != 1 or e.shape != t.shape or not len(t) or not np.isfinite(t).all() or (t <= 0).any() or not np.isin(e, [0,1]).all():
        raise ValueError('Require positive finite times and 0/1 observed-event flags')
    return t, e.astype(bool)


def fit(features, durations, events, *, penalizer=.1):
    require('sklearn','survival')
    from nil._survival import CoxModel
    return CoxModel(penalizer=penalizer).fit(features, (durations, events))


def kaplan_meier(durations, events):
    require('sklearn','survival')
    from nil._survival import KaplanMeier
    return KaplanMeier().fit(durations, events)


def evaluate(model, features, durations, events):
    from nil._classical_common import unwrap
    np = require('numpy', 'classical')
    times, observed = _outcomes(durations, events)
    risk = np.asarray(unwrap(model).predict(features), dtype=float)
    if risk.shape != times.shape or not np.isfinite(risk).all(): raise ValueError('Need one finite risk score per subject')
    try:
        score = require('lifelines.utils', 'survival').concordance_index(times, -risk, observed)
    except ZeroDivisionError as error:
        raise ValueError('No comparable survival pairs') from error
    return {'count': len(times), 'observed_events': int(observed.sum()), 'concordance_index': float(score)}
