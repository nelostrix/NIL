import pytest
import torch
import nil


def test_maml_gradient_matches_closed_form_without_model_mutation():
    model=torch.nn.Linear(1,1,bias=False).double().eval()
    with torch.no_grad():model.weight.fill_(.3)
    x=torch.tensor([[1.],[2.]],dtype=torch.double);y=2*x
    parameters=nil.learning.adapt(model,x,y,torch.nn.functional.mse_loss,lr=.1)
    query=torch.func.functional_call(model,parameters,(x,))
    gradient=torch.autograd.grad(torch.nn.functional.mse_loss(query,y),model.weight)[0]
    # MSE gradient=5(w-2); inner w' = w-.1*5(w-2); d outer/dw=5(w'-2)*.5.
    assert gradient.item()==pytest.approx(5*(1.15-2)*.5)
    assert model.weight.item()==.3 and model.weight.grad is None
    first=nil.learning.adapt(model,x,y,torch.nn.functional.mse_loss,lr=.1,first_order=True)
    first_gradient=torch.autograd.grad(torch.nn.functional.mse_loss(torch.func.functional_call(model,first,(x,)),y),model.weight)[0]
    assert first_gradient.item()==pytest.approx(5*(1.15-2))


@pytest.mark.parametrize('directed',[False,True])
@pytest.mark.parametrize('normalization',['row','symmetric'])
def test_weighted_graph_against_dense_and_gradients(directed,normalization):
    x=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],dtype=torch.double,requires_grad=True)
    edges=torch.tensor([[0,1,1],[1,2,1]]);w=torch.tensor([2.,3.,4.],dtype=torch.double,requires_grad=True)
    result=nil.graphs.weighted_propagate(x,edges,w,directed=directed,normalization=normalization)
    adjacency=torch.zeros(3,3,dtype=torch.double)
    for (a,b),weight in zip(edges.T,w.detach()):
        adjacency[b,a]+=weight
        if not directed and a!=b:adjacency[a,b]+=weight
    for i in range(3):
        if adjacency[i,i]==0:adjacency[i,i]=1
    if normalization=='row':expected=adjacency/adjacency.sum(1)[:,None]
    else:expected=adjacency/(adjacency.sum(1)[:,None]*adjacency.sum(0)[None,:]).sqrt()
    torch.testing.assert_close(result,expected@x)
    result.square().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()


@pytest.mark.parametrize('directed',[False,True])
def test_complement_sampling_exhaustive_small_graphs(directed):
    for n in range(2,9):
        universe={(a,b) for a in range(n) for b in range(n) if a!=b and (directed or a<b)}
        blocked=set(sorted(universe)[::2]);edges=torch.tensor(sorted(blocked),dtype=torch.long).T
        expected=universe-blocked
        if expected:
            negatives=nil.graphs.negative_edges(n,edges,count=len(expected),directed=directed)
            actual=list(map(tuple,negatives.T.tolist()))
            assert len(actual)==len(set(actual)) and set(actual)==expected


def test_link_split_no_leakage_neighbor_sampling_and_weighted_model():
    edges=torch.tensor([[0,1,2,3],[1,2,3,4]])
    split=nil.graphs.split_links(6,edges,seed=3)
    positives=set(map(tuple,edges.T.tolist()))
    negatives=set(map(tuple,torch.cat([split['train_negative'],split['test_negative']],1).T.tolist()))
    assert not positives&negatives
    x=torch.arange(18,dtype=torch.float32).reshape(6,3)
    sampled=nil.graphs.sample_neighbors(x,edges,[2],fanouts=(1,1),seed=4)
    torch.testing.assert_close(sampled['features'],x[sampled['nodes']])
    assert sampled['nodes'][sampled['seed_positions']].tolist()==[2]
    model=nil.graphs.weighted_gcn(3,4,2,directed=True)
    result=model(x,edges,torch.ones(4));result.sum().backward()
    assert result.shape==(6,2) and all(p.grad is not None for p in model.parameters())
    assert nil.graphs.link_logits(result,edges).shape==(4,)


def test_meta_trained_initialization_improves_heldout_adaptation():
    model=torch.nn.Linear(1,1,bias=False).eval()
    with torch.no_grad():model.weight.zero_()
    support=torch.tensor([[1.]]);query=torch.tensor([[2.],[3.]])
    loss=torch.nn.functional.mse_loss;optimizer=torch.optim.SGD(model.parameters(),lr=.05)
    def objective(slopes):
        values=[]
        for slope in slopes:
            adapted=nil.learning.adapt(model,support,support*slope,loss,lr=.1)
            values.append(loss(torch.func.functional_call(model,adapted,(query,)),query*slope))
        return torch.stack(values).mean()
    before=objective([1.7,2.3]).item()
    for _ in range(30):
        optimizer.zero_grad();value=objective([1.5,2.,2.5]);value.backward();optimizer.step()
    after=objective([1.7,2.3]).item()
    assert after<before*.05


def test_link_model_learns_heldout_synthetic_links_without_message_passing_leakage():
    torch.manual_seed(9)
    edges=torch.tensor([(a,b) for a in range(8) for b in range(a+1,8) if a//4==b//4]).T
    split=nil.graphs.split_links(8,edges,test_fraction=.25,seed=9)
    train=split['train_positive'];x=torch.eye(8);weights=torch.ones(train.shape[1])
    model=nil.graphs.weighted_gcn(8,8,4);optimizer=torch.optim.Adam(model.parameters(),lr=.03)
    fit_edges=torch.cat([train,split['train_negative']],1)
    labels=torch.cat([torch.ones(train.shape[1]),torch.zeros(split['train_negative'].shape[1])])
    for _ in range(80):
        optimizer.zero_grad();embedding=model(x,train,weights)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(nil.graphs.link_logits(embedding,fit_edges),labels)
        loss.backward();optimizer.step()
    with torch.no_grad():
        embedding=model(x,train,weights)
        test_edges=torch.cat([split['test_positive'],split['test_negative']],1)
        target=torch.cat([torch.ones(split['test_positive'].shape[1]),torch.zeros(split['test_negative'].shape[1])]).bool()
        accuracy=((nil.graphs.link_logits(embedding,test_edges)>0)==target).float().mean().item()
    assert accuracy>=.8
