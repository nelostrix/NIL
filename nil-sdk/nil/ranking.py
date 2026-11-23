"""Query-safe learning-to-rank with explicit per-query NDCG and average precision."""
from nil.runtime import require
from nil._classical_common import positive_int


def _queries(groups, count):
    np = require('numpy', 'classical')
    values = np.asarray(groups,dtype=object)
    if values.ndim != 1 or len(values) != count or not count:
        raise ValueError('groups must contain one query ID per row')
    queries = {}
    for i, value in enumerate(values.tolist()):
        if value is None or not isinstance(value, (str, int, float)) or isinstance(value, float) and not np.isfinite(value):
            raise ValueError('Query IDs must be finite scalar strings/numbers')
        queries.setdefault((type(value).__name__, value), []).append(i)
    return [np.asarray(indices) for indices in queries.values()]


def evaluate(targets, scores, groups, *, k=10):
    np = require('numpy', 'classical'); positive_int(k, 'k')
    y, score = np.asarray(targets, dtype=float), np.asarray(scores, dtype=float)
    if y.ndim != 1 or score.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(score).all() or (y < 0).any() or (y > 31).any():
        raise ValueError('Ranking requires finite scores and relevance in [0,31]')
    ndcg, average_precision = [], []
    for indices in _queries(groups, len(y)):
        actual, predicted = y[indices], score[indices]
        order = np.argsort(-predicted, kind='stable')[:k]
        ideal = np.sort(actual)[::-1][:k]
        discount = np.log2(np.arange(len(order))+2)
        denominator = ((2**ideal-1)/discount).sum()
        ndcg.append(float(((2**actual[order]-1)/discount).sum()/denominator) if denominator else 0.0)
        relevant = actual[order] > 0
        denominator = min(int((actual > 0).sum()), k)
        average_precision.append(float(((np.cumsum(relevant)/(np.arange(len(order))+1))*relevant).sum()/denominator) if denominator else 0.0)
    return {'queries': len(ndcg), f'ndcg@{k}': float(np.mean(ndcg)), f'map@{k}': float(np.mean(average_precision))}


def fit(features, relevance, groups, *, estimator=None, sample_weight=None):
    require('sklearn','lightgbm')
    from nil._ranking import QueryRanker
    return QueryRanker(estimator=estimator).fit(features, relevance, groups=groups, sample_weight=sample_weight)


def split(features, relevance, groups, *, test_size=.2, seed=0):
    from nil.validation import split as partition
    np=require('numpy','classical'); normalized=np.empty(len(relevance),dtype=int)
    for index,rows in enumerate(_queries(groups,len(relevance))): normalized[rows]=index
    return partition(features, relevance, groups=normalized, method='group', test_size=test_size, seed=seed)
