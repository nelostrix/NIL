"""Bounded sparse item-cosine retrieval and popularity cold starts."""
def fit(interactions, *, neighbors=20, max_items=10000, max_pairs=2000000):
    from nil.runtime import require
    require('sklearn','classical')
    from nil._recommendation import ItemRecommender
    return ItemRecommender(neighbors=neighbors, max_items=max_items, max_pairs=max_pairs).fit(interactions)


def evaluate(model, heldout, *, k=10, max_users=10000):
    from nil.runtime import require
    from nil._classical_common import positive_int
    np=require('numpy','classical'); sparse=require('scipy.sparse','classical')
    positive_int(k,'k'); positive_int(max_users,'max_users')
    if not sparse.issparse(heldout): raise TypeError('heldout must be a sparse user/item matrix')
    actual=sparse.csr_matrix(heldout,dtype=float,copy=True); actual.sum_duplicates(); actual.eliminate_zeros()
    if actual.shape!=model.interactions_.shape or actual.shape[0]>max_users or not np.isfinite(actual.data).all() or (actual.data<0).any(): raise ValueError('Held-out schema/data/user budget differs')
    if actual.multiply(model.interactions_).nnz: raise ValueError('Training/held-out interactions must be disjoint')
    precision=[]; recall=[]; ndcg=[]
    for user in range(actual.shape[0]):
        relevant=set(actual.getrow(user).indices)
        if not relevant: continue
        recommended=model.recommend(user,k=k); hits=np.asarray([item in relevant for item,_ in recommended],dtype=float)
        precision.append(float(hits.sum()/k)); recall.append(float(hits.sum()/len(relevant)))
        ideal=np.sum(1/np.log2(np.arange(min(k,len(relevant)))+2))
        ndcg.append(float(np.sum(hits/np.log2(np.arange(len(hits))+2))/ideal))
    if not precision: raise ValueError('No held-out interactions to evaluate')
    return {'users':len(precision),f'precision@{k}':float(np.mean(precision)),f'recall@{k}':float(np.mean(recall)),f'ndcg@{k}':float(np.mean(ndcg))}
