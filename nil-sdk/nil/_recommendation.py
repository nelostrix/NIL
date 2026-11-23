import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from nil._classical_common import positive_int


class ItemRecommender(BaseEstimator):
    def __init__(self, *, neighbors=20, max_items=10000, max_pairs=2000000):
        self.neighbors, self.max_items, self.max_pairs = neighbors, max_items, max_pairs

    def fit(self, X, y=None):
        for name in ('neighbors', 'max_items', 'max_pairs'): positive_int(getattr(self, name), name)
        if not sparse.issparse(X): raise TypeError('Use an explicit scipy sparse user/item matrix')
        values = sparse.csr_matrix(X, dtype=float, copy=True); values.sum_duplicates(); values.eliminate_zeros()
        if not values.shape[0] or not values.shape[1] or values.shape[1] > self.max_items or not np.isfinite(values.data).all() or (values.data < 0).any():
            raise ValueError('Interactions must be finite, nonnegative and within max_items')
        degree = np.diff(values.indptr)
        if sum(int(v)**2 for v in degree) > self.max_pairs:
            raise ValueError('Potential sparse co-occurrences exceed max_pairs before multiplication')
        binary = values.copy(); binary.data[:] = 1
        gram = (binary.T@binary).tocsr()
        norms = np.sqrt(np.asarray(binary.sum(axis=0)).reshape(-1))
        rows, cols = gram.nonzero()
        gram.data /= norms[rows]*norms[cols]
        gram.setdiag(0); gram.eliminate_zeros()
        for i in range(gram.shape[0]):
            start, stop = gram.indptr[i:i+2]
            if stop-start > self.neighbors:
                order = np.lexsort((gram.indices[start:stop], -gram.data[start:stop]))
                gram.data[start+order[self.neighbors:]] = 0
        gram.eliminate_zeros()
        self.interactions_, self.similarities_ = values, gram
        self.popularity_, self.n_features_in_ = np.asarray(binary.sum(axis=0)).reshape(-1), values.shape[1]
        return self

    def recommend(self, user=None, *, k=10, exclude_seen=True):
        check_is_fitted(self, 'similarities_'); positive_int(k, 'k')
        if type(exclude_seen) is not bool: raise ValueError('exclude_seen must be boolean')
        if k > self.max_items: raise ValueError('k exceeds max_items')
        seen = np.array([], dtype=int)
        if user is None:
            scores = self.popularity_.copy()
        else:
            if type(user) is not int or not 0 <= user < self.interactions_.shape[0]:
                raise ValueError('Use a known user row index or None for cold start')
            row = self.interactions_.getrow(user); seen = row.indices
            scores = np.asarray((row@self.similarities_).toarray()).reshape(-1)
            if not row.nnz or not (scores > 0).any(): scores = self.popularity_.copy()
        if exclude_seen: scores[seen] = -np.inf
        order = np.lexsort((np.arange(len(scores)), -scores))
        return [(int(i), float(scores[i])) for i in order[:k] if np.isfinite(scores[i])]

    def predict(self, users):
        return [self.recommend(user) for user in users]
