"""Graph convolution parity, disjoint batching and actual graph/contrastive learning."""
import pytest
torch=pytest.importorskip('torch')
torch.set_num_threads(1)
import nil


def test_graph_normalization_matches_dense_reference_and_gradients():
    x=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],requires_grad=True)
    edges=torch.tensor([[0,1,1,1],[1,0,2,2]])
    adjacency=torch.tensor([[1.,1.,0.],[1.,1.,1.],[0.,1.,1.]])
    degree=adjacency.sum(1);normalized=adjacency/torch.sqrt(degree[:,None]*degree[None,:])
    actual=nil.graphs.propagate(x,edges)
    torch.testing.assert_close(actual,normalized@x)
    actual.sum().backward();torch.testing.assert_close(x.grad,normalized.T@torch.ones_like(x))
    model=nil.graphs.gcn(2,3,2)
    expected=model.second(normalized@model.first(normalized@x).relu())
    torch.testing.assert_close(model(x,edges),expected)


def test_graph_batch_has_no_cross_graph_edges_and_dtype_reload(tmp_path):
    first=torch.tensor([[1.],[2.]],dtype=torch.float64);second=torch.tensor([[10.],[20.]],dtype=torch.float64)
    edges=torch.tensor([[0],[1]])
    packed=nil.graphs.batch([(first,edges),(second,edges)])
    output=nil.graphs.propagate(packed['features'],packed['edge_index'])
    torch.testing.assert_close(output[:2],nil.graphs.propagate(first,edges))
    torch.testing.assert_close(output[2:],nil.graphs.propagate(second,edges))
    torch.testing.assert_close(nil.graphs.pool(output,packed['batch']),torch.tensor([[1.5],[15.]],dtype=torch.float64))
    restored=nil.graphs.load(nil.graphs.save(first,edges,tmp_path/'graph'))
    assert restored['features'].dtype==torch.float64
    torch.testing.assert_close(restored['features'],first,rtol=0,atol=0)
    with pytest.raises(ValueError):nil.graphs.propagate(first,torch.tensor([[0],[5]]))
    with pytest.raises(ValueError):nil.graphs.pool(first,torch.tensor([0,2]))


def test_node_classification_actual_learning_and_checkpoint(tmp_path):
    torch.manual_seed(7)
    labels=torch.arange(40)%2
    features=torch.stack([labels.float()*2-1,torch.randn(40)*.1],1)
    pairs=[(i,j) for i in range(40) for j in range(i+1,40) if labels[i]==labels[j] and (i+j)%5==0]
    edges=torch.tensor(pairs).T.contiguous()
    model=nil.graphs.gcn(2,8,2)
    # One graph is transductive; labels outside the training mask never enter loss.
    training=torch.arange(40)<20;heldout=~training
    def task(network,batch):
        x,e,y,mask=batch[0]
        return torch.nn.functional.cross_entropy(network(x,e)[mask],y[mask])
    data=nil.data.Records([(features,edges,labels,training)])
    # Use a user loader-like list so tensors have no artificial batch dimension.
    class Step:
        def __call__(self,network,batch):return torch.nn.functional.cross_entropy(network(features,edges)[training],labels[training])
    result=nil.train(model,nil.data.tensors(torch.zeros(1,1),torch.zeros(1,1)),task=Step(),epochs=35,batch_size=1,learning_rate=.03,device='cpu',output_dir=tmp_path)
    logits=nil.predict(result,dict(features=features,edge_index=edges),device_name='cpu')
    assert (logits[heldout].argmax(1)==labels[heldout]).float().mean()>.9
    restored=nil.graphs.gcn(2,8,2);nil.training.checkpoint.load(result.checkpoint,restored)
    torch.testing.assert_close(nil.predict(restored,dict(features=features,edge_index=edges),device_name='cpu'),logits,rtol=0,atol=0)


def test_graph_classifier_trains_with_inductive_heldout_graphs(tmp_path):
    torch.manual_seed(4)
    graphs=[];labels=[]
    for i in range(20):
        label=i%2;features=torch.randn(4,2)*.1+(2*label-1)
        graphs.append((features,torch.tensor([[0,1,2],[1,2,3]])));labels.append(label)
    train=nil.graphs.batch(graphs[:12]);valid=nil.graphs.batch(graphs[12:]);targets=torch.tensor(labels)
    model=nil.graphs.gcn(2,4,2,task='graph');optimizer=torch.optim.Adam(model.parameters(),lr=.05)
    for _ in range(30):
        optimizer.zero_grad();torch.nn.functional.cross_entropy(model(**train),targets[:12]).backward();optimizer.step()
    assert (nil.predict(model,valid,device_name='cpu').argmax(1)==targets[12:]).float().mean()==1


def test_contrastive_loss_reference_gradient_and_optimization():
    torch.manual_seed(2)
    a=torch.randn(8,3,requires_grad=True);b=(a.detach()+.1*torch.randn(8,3)).requires_grad_()
    z=torch.nn.functional.normalize(torch.cat([a,b]),dim=1);scores=z@z.T/.2
    numerator=torch.exp(scores[torch.arange(16),(torch.arange(16)+8)%16])
    denominator=torch.stack([torch.exp(torch.cat([scores[i,:i],scores[i,i+1:]])).sum() for i in range(16)])
    expected=-(numerator/denominator).log().mean()
    loss=nil.learning.contrastive_loss(a,b);torch.testing.assert_close(loss,expected)
    loss.backward();assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    encoder=torch.nn.Linear(3,4);optimizer=torch.optim.Adam(encoder.parameters(),lr=.03)
    x=torch.randn(16,3);view=x+.03*torch.randn(16,3)
    before=nil.learning.contrastive_loss(encoder(x),encoder(view)).item()
    for _ in range(40):
        optimizer.zero_grad();objective=nil.learning.contrastive_loss(encoder(x),encoder(view));objective.backward();optimizer.step()
    assert nil.learning.contrastive_loss(encoder(x),encoder(view)).item()<before
    with pytest.raises(ValueError):nil.learning.contrastive_loss(torch.zeros(3,2),torch.ones(3,2))
