"""Validated homogeneous undirected graphs and opt-in Torch GCN workflows."""
from nil.runtime import require
from nil._workflow_io import save as _save, load as _load


def validate(features, edge_index):
    torch=require('torch','torch')
    x=torch.as_tensor(features);edges=torch.as_tensor(edge_index)
    if x.ndim!=2 or not 1<=len(x)<=100000 or not 1<=x.shape[1]<=4096 or x.numel()>4000000 or x.dtype not in (torch.float32,torch.float64) or not torch.isfinite(x).all():raise ValueError('Finite floating [nodes,features] required')
    if edges.ndim!=2 or edges.shape[0]!=2 or edges.shape[1]>1000000 or edges.dtype!=torch.int64 or edges.device!=x.device or (edges<0).any() or (edges>=len(x)).any():raise ValueError('int64 [2,edges] in-range endpoints on feature device required')
    return x,edges


def propagate(features, edge_index):
    """D^-1/2(A+I)D^-1/2 X; symmetrize/deduplicate and add exactly one self loop."""
    torch=require('torch','torch');x,edges=validate(features,edge_index)
    nodes=torch.arange(len(x),device=x.device)
    all_edges=torch.cat([edges,edges.flip(0),torch.stack([nodes,nodes])],dim=1)
    # Integer encoding provides stable deduplication without a dense adjacency.
    codes=torch.unique(all_edges[0]*len(x)+all_edges[1],sorted=True)
    source=torch.div(codes,len(x),rounding_mode='floor');target=codes%len(x)
    degree=torch.bincount(target,minlength=len(x)).to(x.dtype)
    weights=(degree[source]*degree[target]).rsqrt()
    output=torch.zeros_like(x);output.index_add_(0,target,x[source]*weights[:,None])
    return output


def pool(features, batch):
    torch=require('torch','torch');x=torch.as_tensor(features);ids=torch.as_tensor(batch)
    if x.ndim!=2 or not len(x) or not x.is_floating_point() or not torch.isfinite(x).all() or ids.dtype!=torch.int64 or ids.device!=x.device or ids.shape!=(len(x),) or (ids<0).any():raise ValueError('Invalid graph pooling inputs')
    count=int(ids.max())+1
    if count>len(x):raise ValueError('Graph IDs must be contiguous and nonempty')
    sizes=torch.bincount(ids,minlength=count)
    if (sizes==0).any():raise ValueError('Graph IDs must be contiguous and nonempty')
    result=torch.zeros((count,x.shape[1]),device=x.device,dtype=x.dtype)
    result.index_add_(0,ids,x);return result/sizes[:,None]


def batch(graphs):
    torch=require('torch','torch');xs=[];edges=[];ids=[];offset=0;width=None;device=None
    for i,(x,e) in enumerate(graphs):
        if i>=1000:raise ValueError('Too many graphs')
        x,e=validate(x,e)
        if width is not None and (x.shape[1]!=width or x.device!=device or x.dtype!=xs[0].dtype):raise ValueError('Graph feature dimensions/device/dtype must match')
        if offset+len(x)>100000 or sum(v.shape[1] for v in edges)+e.shape[1]>1000000:raise ValueError('Graph batch exceeds budget')
        width=x.shape[1];device=x.device;xs.append(x);edges.append(e+offset);ids.append(torch.full((len(x),),i,dtype=torch.long,device=device));offset+=len(x)
    if not xs:raise ValueError('Empty graph batch')
    return dict(features=torch.cat(xs),edge_index=torch.cat(edges,dim=1),batch=torch.cat(ids))


def gcn(input_dim, hidden_dim, output_dim, *, task='node'):
    for n in (input_dim,hidden_dim,output_dim):
        if type(n) is not int or not 1<=n<=4096:raise ValueError('Positive bounded GCN dimensions required')
    if input_dim*hidden_dim+hidden_dim*output_dim>4000000 or task not in {'node','graph'}:raise ValueError('GCN parameter/task budget exceeded')
    from nil._graph_models import GCN
    return GCN(input_dim,hidden_dim,output_dim,task)


def save(features,edge_index,path):
    x,e=validate(features,edge_index)
    return _save(path,'graph',dict(features=x.detach().cpu().tolist(),edge_index=e.cpu().tolist(),dtype=str(x.dtype).split('.')[-1]))


def load(path):
    torch=require('torch','torch');r=_load(path,'graph')
    if r.get('dtype') not in {'float32','float64'}:raise ValueError('Unsupported graph dtype')
    x=torch.tensor(r['features'],dtype=getattr(torch,r['dtype']));e=torch.tensor(r['edge_index'],dtype=torch.int64)
    validate(x,e);return dict(features=x,edge_index=e)


def weighted_propagate(features,edge_index,edge_weight,*,directed=False,normalization='symmetric',self_loops=True):
    """Weighted COO coalescing; row or out/in symmetric normalization.

    Weighted duplicate edges sum. Undirected mode adds reverse edges; supply each
    unordered edge once to avoid doubling a supplied bidirectional pair.
    Existing self-loop weights are preserved; missing loops receive weight1.
    """
    torch=require('torch','torch');x,edges=validate(features,edge_index)
    weights=torch.as_tensor(edge_weight,device=x.device)
    if weights.shape!=(edges.shape[1],) or weights.dtype!=x.dtype or not torch.isfinite(weights).all() or (weights<0).any() or (weights>1e6).any():raise ValueError('Finite nonnegative matching-dtype edge weights required')
    if type(directed) is not bool or type(self_loops) is not bool or normalization not in {'row','symmetric'}:raise ValueError('Invalid graph normalization')
    if not directed:
        nonloops=edges[0]!=edges[1]
        weights=torch.cat([weights,weights[nonloops]])
        edges=torch.cat([edges,edges[:,nonloops].flip(0)],dim=1)
    codes=edges[0]*len(x)+edges[1]
    if self_loops:
        present=torch.zeros(len(x),dtype=torch.bool,device=x.device)
        present[edges[0,edges[0]==edges[1]]]=True
        missing=torch.arange(len(x),device=x.device)[~present]
        codes=torch.cat([codes,missing*len(x)+missing]);weights=torch.cat([weights,torch.ones(len(missing),dtype=x.dtype,device=x.device)])
    unique,inverse=torch.unique(codes,sorted=True,return_inverse=True)
    combined=torch.zeros(len(unique),dtype=x.dtype,device=x.device).index_add(0,inverse,weights)
    source=torch.div(unique,len(x),rounding_mode='floor');target=unique%len(x)
    incoming=torch.zeros(len(x),dtype=x.dtype,device=x.device).index_add(0,target,combined)
    tiny=torch.finfo(x.dtype).tiny
    if normalization=='row':scale=combined/incoming[target].clamp_min(tiny)
    else:
        outgoing=torch.zeros(len(x),dtype=x.dtype,device=x.device).index_add(0,source,combined)
        scale=combined/(outgoing[source]*incoming[target]).clamp_min(tiny).sqrt()
    return torch.zeros_like(x).index_add(0,target,x[source]*scale[:,None])


def weighted_gcn(input_dim,hidden_dim,output_dim,*,directed=False,normalization='symmetric',task='node'):
    for n in (input_dim,hidden_dim,output_dim):
        if type(n) is not int or not 1<=n<=4096:raise ValueError('Positive bounded GCN dimensions required')
    if input_dim*hidden_dim+hidden_dim*output_dim>4000000 or task not in {'node','graph'} or type(directed) is not bool or normalization not in {'row','symmetric'}:raise ValueError('Invalid weighted GCN configuration')
    from nil._graph_models import WeightedGCN
    return WeightedGCN(input_dim,hidden_dim,output_dim,task,directed,normalization)


def negative_edges(num_nodes,positive_edges,*,count,seed=0,directed=False):
    """Exact bounded sampling without replacement from the true non-edge set.

    Uses rank selection over an implicit universe, not dense adjacency or an
    unbounded rejection loop. Self edges are excluded; all held-out positives
    should be supplied so they cannot become false negatives.
    """
    import bisect,random
    torch=require('torch','torch')
    if type(num_nodes) is not int or not 2<=num_nodes<=100000 or type(count) is not int or not 1<=count<=1000000 or type(seed) is not int or not 0<=seed<2**32 or type(directed) is not bool:raise ValueError('Invalid negative-edge budget')
    edges=torch.as_tensor(positive_edges)
    if edges.ndim!=2 or edges.shape[0]!=2 or edges.dtype!=torch.int64 or edges.shape[1]>1000000 or edges.device.type!='cpu' or (edges<0).any() or (edges>=num_nodes).any():raise ValueError('Bounded CPU int64 endpoints required')
    def rank(a,b):
        if directed:return a*(num_nodes-1)+b-(b>a)
        if a>b:a,b=b,a
        return a*(2*num_nodes-a-1)//2+b-a-1
    blocked=sorted({rank(a,b) for a,b in edges.T.tolist() if a!=b})
    total=num_nodes*(num_nodes-1)//(1 if directed else 2);available=total-len(blocked)
    if count>available:raise ValueError('Not enough true nonedges for requested count')
    choices=random.Random(seed).sample(range(available),count);pairs=[]
    for value in choices:
        low=value;high=value+len(blocked)
        while low<high:
            middle=(low+high)//2
            if middle+1-bisect.bisect_right(blocked,middle)>=value+1:high=middle
            else:low=middle+1
        index=low
        if directed:
            a,b=divmod(index,num_nodes-1)
            if b>=a:b+=1
        else:
            low=0;high=num_nodes-1
            while low<high:
                middle=(low+high+1)//2
                if middle*(2*num_nodes-middle-1)//2<=index:low=middle
                else:high=middle-1
            a=low;b=a+1+index-a*(2*num_nodes-a-1)//2
        pairs.append((a,b))
    return torch.tensor(pairs,dtype=torch.long).T.contiguous()


def split_links(num_nodes,positive_edges,*,test_fraction=.2,seed=0,directed=False):
    torch=require('torch','torch')
    if type(test_fraction) not in (int,float) or not 0<test_fraction<1:raise ValueError('Invalid edge test fraction')
    edges=torch.as_tensor(positive_edges)
    # Validate before canonicalizing and reserve a balanced negative sample.
    negative_edges(num_nodes,edges,count=1,seed=seed,directed=directed)
    pairs=sorted({(a,b) if directed or a<b else (b,a) for a,b in edges.T.tolist() if a!=b})
    if len(pairs)<2:raise ValueError('At least two distinct positive links required')
    import random
    random.Random(seed).shuffle(pairs);test=max(1,min(len(pairs)-1,round(len(pairs)*test_fraction)))
    positives=torch.tensor(pairs,dtype=torch.long).T.contiguous()
    negatives=negative_edges(num_nodes,positives,count=len(pairs),seed=seed+1 if seed<2**32-1 else 0,directed=directed)
    return dict(train_positive=positives[:,test:],test_positive=positives[:,:test],train_negative=negatives[:,test:],test_negative=negatives[:,:test])


def link_logits(embeddings,edges):
    torch=require('torch','torch');x,e=validate(embeddings,edges)
    return (x[e[0]]*x[e[1]]).sum(1)


def sample_neighbors(features,edge_index,seeds,*,fanouts=(10,5),seed=0,max_nodes=10000,directed=False):
    """Uniform CPU incoming-neighbor sampling; not an unbiased full-GCN estimate."""
    import random
    torch=require('torch','torch');x,e=validate(features,edge_index)
    if x.device.type!='cpu' or type(seed) is not int or not 0<=seed<2**32 or type(max_nodes) is not int or not 1<=max_nodes<=100000 or type(directed) is not bool or not isinstance(fanouts,(tuple,list)) or not 1<=len(fanouts)<=3 or any(type(k) is not int or not 1<=k<=64 for k in fanouts):raise ValueError('Invalid CPU sampling configuration')
    ids=list(seeds)
    if not 1<=len(ids)<=512 or any(type(i) is not int or not 0<=i<len(x) for i in ids) or len(set(ids))!=len(ids) or len(ids)>max_nodes:raise ValueError('Invalid unique seed nodes')
    adjacency={}
    for a,b in e.T.tolist():
        adjacency.setdefault(b,set()).add(a)
        if not directed:adjacency.setdefault(a,set()).add(b)
    nodes=list(ids);known=set(ids);frontier=ids;rng=random.Random(seed);selected=set()
    for width in fanouts:
        following=[]
        for target in frontier:
            neighbors=sorted(adjacency.get(target,()))
            chosen=rng.sample(neighbors,min(width,len(neighbors)))
            for source in chosen:
                selected.add((source,target))
                if source not in known:
                    if len(nodes)>=max_nodes:raise ValueError('Sampled graph exceeds node budget')
                    nodes.append(source);known.add(source);following.append(source)
        frontier=following
    lookup={node:i for i,node in enumerate(nodes)}
    sampled=torch.tensor([(lookup[a],lookup[b]) for a,b in sorted(selected)],dtype=torch.long).T if selected else torch.empty((2,0),dtype=torch.long)
    return dict(features=x[nodes].clone(),edge_index=sampled,nodes=torch.tensor(nodes),seed_positions=torch.arange(len(ids)))


def _hetero_validate(features,edges):
    torch=require('torch','torch')
    if not isinstance(features,dict) or not 1<=len(features)<=16 or not isinstance(edges,dict) or len(edges)>64:raise ValueError('Require bounded node-type and relation dictionaries')
    device=None;dtype=None;elements=0;nodes=0;edge_count=0
    for name,x in features.items():
        if not isinstance(name,str) or not name or len(name)>128 or not isinstance(x,torch.Tensor) or x.ndim!=2 or not 1<=len(x)<=100000 or not 1<=x.shape[1]<=4096 or x.dtype not in (torch.float32,torch.float64) or not torch.isfinite(x).all():raise ValueError('Invalid typed node features')
        if device is not None and (x.device!=device or x.dtype!=dtype):raise ValueError('Node types must share device/dtype')
        device=x.device;dtype=x.dtype;elements+=x.numel();nodes+=len(x)
    if elements>1000000 or nodes>100000:raise ValueError('Heterogeneous feature budget exceeded')
    for relation,e in edges.items():
        if not isinstance(relation,tuple) or len(relation)!=3 or any(not isinstance(s,str) or not s or len(s)>128 for s in relation) or relation[0] not in features or relation[2] not in features:raise ValueError('Relation requires (source_type,name,target_type)')
        if not isinstance(e,torch.Tensor) or e.dtype!=torch.int64 or e.device!=device or e.ndim!=2 or e.shape[0]!=2 or (e<0).any() or (e[0]>=len(features[relation[0]])).any() or (e[1]>=len(features[relation[2]])).any():raise ValueError('Invalid typed relation endpoints')
        edge_count+=e.shape[1]
        if edge_count>1000000:raise ValueError('Heterogeneous edge budget exceeded')
    return features,edges


def relation_mean(source,edge_index,target_nodes):
    """Incoming neighbor mean for bipartite COO edges; repeated edges count twice."""
    torch=require('torch','torch')
    if type(target_nodes) is not int or not 1<=target_nodes<=100000:raise ValueError('Invalid target count')
    target=source.new_empty((target_nodes,1)) if isinstance(source,torch.Tensor) else None
    if target is None:raise ValueError('Torch source features required')
    # No target feature values are read: use finite zeros for shared validation.
    target.zero_();_hetero_validate({'source':source,'target':target},{('source','to','target'):edge_index})
    if target_nodes*source.shape[1]>1000000:raise ValueError('Relation message budget exceeded')
    result=source.new_zeros((target_nodes,source.shape[1]))
    result.index_add_(0,edge_index[1],source[edge_index[0]])
    degree=torch.bincount(edge_index[1],minlength=target_nodes).to(source.dtype).clamp_min(1)
    return result/degree[:,None]


def hetero_sage(input_dims,hidden_dim,output_dims,relations):
    """Two-layer typed self + relation-specific incoming-mean neural network."""
    if not isinstance(input_dims,dict) or not isinstance(output_dims,dict) or set(input_dims)!=set(output_dims) or not 1<=len(input_dims)<=16 or any(not isinstance(k,str) or not k or len(k)>128 for k in input_dims):raise ValueError('Matching bounded node-type dimensions required')
    if type(hidden_dim) is not int or not 1<=hidden_dim<=512 or any(type(v) is not int or not 1<=v<=4096 for v in [*input_dims.values(),*output_dims.values()]):raise ValueError('Invalid heterogeneous model dimensions')
    if not isinstance(relations,(list,tuple)) or not 1<=len(relations)<=64 or any(not isinstance(r,tuple) or len(r)!=3 or any(not isinstance(s,str) or not s or len(s)>128 for s in r) or r[0] not in input_dims or r[2] not in input_dims for r in relations) or len(set(relations))!=len(relations):raise ValueError('Unique declared relations required')
    relations=tuple(sorted(relations));types=sorted(input_dims)
    count=sum((input_dims[t]+1)*hidden_dim+(hidden_dim+1)*output_dims[t] for t in types)+sum(input_dims[a]*hidden_dim+hidden_dim*output_dims[b] for a,_,b in relations)
    if count>1000000:raise ValueError('Heterogeneous parameter budget exceeds 1M')
    from nil._graph_models import HeteroSAGE
    return HeteroSAGE(dict(input_dims),hidden_dim,dict(output_dims),relations)


def save_hetero(features,edges,path):
    x,e=_hetero_validate(features,edges)
    return _save(path,'heterogeneous-graph',dict(features={k:v.detach().cpu().tolist() for k,v in x.items()},
        dtype=str(next(iter(x.values())).dtype).split('.')[-1],relations=[dict(type=list(k),edge_index=v.cpu().tolist()) for k,v in sorted(e.items())]))


def load_hetero(path):
    torch=require('torch','torch');r=_load(path,'heterogeneous-graph')
    if r.get('dtype') not in ('float32','float64') or not isinstance(r.get('relations'),list) or len(r['relations'])>64:raise ValueError('Invalid heterogeneous snapshot')
    features={k:torch.tensor(v,dtype=getattr(torch,r['dtype'])) for k,v in r['features'].items()};edges={}
    for row in r['relations']:
        key=tuple(row['type'])
        if key in edges:raise ValueError('Duplicate snapshot relation')
        edges[key]=torch.tensor(row['edge_index'],dtype=torch.long)
    _hetero_validate(features,edges);return dict(features=features,edges=edges)
