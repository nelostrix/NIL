"""Explicit active, semi/self-supervised, few-shot and bounded replay primitives."""
import hashlib
import json
import math
from nil.runtime import require
from nil._workflow_io import save as _save, load as _load


def query_uncertain(model, pool, *, budget=1, excluded=()):
    """Deterministic entropy-ranked pool indices. Labels remain oracle-owned."""
    np=require('numpy','classical')
    if type(budget) is not int or not 1<=budget<=10000:raise ValueError('Invalid label budget')
    p=np.asarray(model.predict_proba(pool),dtype=float)
    if p.ndim!=2 or not 1<=len(p)<=100000 or p.shape[1]<2 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):raise ValueError('Expected finite normalized class probabilities')
    excluded=set(excluded)
    if any(type(i) is not int or not 0<=i<len(p) for i in excluded):raise ValueError('Invalid excluded indices')
    if budget>len(p)-len(excluded):raise ValueError('Budget exceeds remaining unlabeled pool')
    entropy=-(p*np.log(np.maximum(p,1e-300))).sum(1)
    return sorted((i for i in range(len(p)) if i not in excluded),key=lambda i:(-entropy[i],i))[:budget]


def self_training(estimator, features, labels, *, threshold=.9, max_iter=10):
    """Fit a cloned sklearn self-training classifier; -1 means unlabeled."""
    np=require('numpy','classical');module=require('sklearn.semi_supervised','classical')
    from sklearn.base import clone
    import inspect
    y=np.asarray(labels)
    if y.ndim!=1 or len(y)!=len(features) or not np.issubdtype(y.dtype,np.integer) or not (y==-1).any() or len(np.unique(y[y!=-1]))<2:raise ValueError('Integer labels, -1 unlabeled rows and two labeled classes required')
    if type(threshold) not in (int,float) or not 0<threshold<1 or type(max_iter) is not int or not 1<=max_iter<=100:raise ValueError('Invalid self-training budget')
    parameter='estimator' if 'estimator' in inspect.signature(module.SelfTrainingClassifier).parameters else 'base_estimator'
    return module.SelfTrainingClassifier(**{parameter:clone(estimator)},threshold=threshold,max_iter=max_iter).fit(features,y)


def prototypes(support, labels, queries):
    """Nearest class mean in caller-provided embedding space; no query-label fit."""
    np=require('numpy','classical');x=np.asarray(support,dtype=float);q=np.asarray(queries,dtype=float);y=np.asarray(labels)
    if x.ndim!=2 or q.ndim!=2 or q.shape[1]!=x.shape[1] or y.shape!=(len(x),) or not len(x) or max(len(x),len(q))>10000 or x.shape[1]>4096 or not np.isfinite(x).all() or not np.isfinite(q).all():raise ValueError('Invalid support/query embeddings')
    if y.dtype.kind not in 'biufUS' or (y.dtype.kind in 'f' and not np.isfinite(y).all()):raise ValueError('Finite numeric or string labels required')
    classes=np.unique(y)
    if len(classes)>1000:raise ValueError('Too many classes')
    means=np.stack([x[y==label].mean(0) for label in classes])
    # Squared-distance expansion avoids an N-query x N-class x feature temporary.
    distances=np.maximum(0,(q*q).sum(1)[:,None]+(means*means).sum(1)[None,:]-2*q@means.T)
    return dict(labels=classes[distances.argmin(1)],classes=classes,distances=distances)


def contrastive_loss(first, second, *, temperature=.2):
    """Symmetric two-view NT-Xent, excluding self similarity; bounded dense logits."""
    torch=require('torch','torch')
    if type(temperature) not in (int,float) or not math.isfinite(temperature) or temperature<=0:raise ValueError('Invalid temperature')
    if not isinstance(first,torch.Tensor) or not isinstance(second,torch.Tensor) or first.ndim!=2 or first.shape!=second.shape or not 2<=len(first)<=512 or not first.is_floating_point() or not second.is_floating_point() or not torch.isfinite(first).all() or not torch.isfinite(second).all() or (first.norm(dim=1)==0).any() or (second.norm(dim=1)==0).any():raise ValueError('Two finite nonzero matching [2..512,features] views required')
    n=len(first);z=torch.nn.functional.normalize(torch.cat([first,second]),dim=1)
    logits=z@z.T/temperature
    logits=logits.masked_fill(torch.eye(2*n,dtype=torch.bool,device=z.device),-torch.inf)
    targets=(torch.arange(2*n,device=z.device)+n)%(2*n)
    return torch.nn.functional.cross_entropy(logits,targets)


class ReplayMemory:
    """Bounded deterministic reservoir of JSON records, with resumable seen count."""
    def __init__(self, capacity=1000, *, seed=0):
        if type(capacity) is not int or not 1<=capacity<=100000 or type(seed) is not int or not 0<=seed<2**32:raise ValueError('Invalid replay capacity/seed')
        self.capacity=capacity;self.seed=seed;self.seen=0;self._records=[]

    def add(self, record):
        if self.seen >= 2**63-1:raise ValueError('Replay counter exhausted')
        text=json.dumps(record,sort_keys=True,allow_nan=False)
        if len(text.encode())>4096:raise ValueError('Replay record exceeds 4 KiB')
        record=json.loads(text);self.seen+=1
        if len(self._records)<self.capacity:self._records.append(record)
        else:
            # Rejection sampling avoids modulo bias for a uniform counter-derived draw.
            limit=(1<<256)-((1<<256)%self.seen);attempt=0
            while True:
                value=int.from_bytes(hashlib.sha256(f'{self.seed}:{self.seen}:{attempt}'.encode()).digest(),'big')
                if value<limit:break
                attempt+=1
            index=value%self.seen
            if index<self.capacity:self._records[index]=record

    def records(self):return json.loads(json.dumps(self._records))
    def save(self,path):return _save(path,'replay-memory',dict(capacity=self.capacity,seed=self.seed,seen=self.seen,records=self._records))

    @classmethod
    def load(cls,path):
        r=_load(path,'replay-memory');memory=cls(r['capacity'],seed=r['seed'])
        if type(r['seen']) is not int or not 0<=r['seen']<2**63 or not isinstance(r['records'],list) or len(r['records'])!=min(r['seen'],memory.capacity):raise ValueError('Invalid replay state')
        for row in r['records']:
            if len(json.dumps(row,allow_nan=False).encode())>4096:raise ValueError('Invalid replay record')
        memory.seen=r['seen'];memory._records=r['records'];return memory
