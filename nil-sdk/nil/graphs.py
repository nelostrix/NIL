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
