"""Typed relation references, actual learning and artifact/schema parity."""
import copy
import pytest
import torch
import nil


RELATION=('author','writes','paper')


def graph():
    return dict(features={'author':torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]]),
                          'paper':torch.zeros(8,1)},
                edges={RELATION:torch.tensor([[0,1,2,3,0,1,2,3],list(range(8))])})


def test_bipartite_mean_matches_dense_reference_and_gradient():
    x=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],dtype=torch.double,requires_grad=True)
    e=torch.tensor([[0,1,1,2],[0,0,0,1]])
    actual=nil.graphs.relation_mean(x,e,3)
    adjacency=torch.tensor([[1.,2.,0.],[0.,0.,1.],[0.,0.,0.]],dtype=torch.double)
    expected=adjacency@x/adjacency.sum(1).clamp_min(1)[:,None]
    torch.testing.assert_close(actual,expected)
    grad=torch.autograd.grad(actual.square().sum(),x)[0]
    reference=torch.autograd.grad(expected.square().sum(),x)[0]
    torch.testing.assert_close(grad,reference)
    assert actual[-1].tolist()==[0.,0.]
    with pytest.raises(ValueError):nil.graphs.relation_mean(x,torch.tensor([[3],[0]]),3)


def test_heterogeneous_node_learning_heldout_labels_and_exact_reload(tmp_path):
    torch.manual_seed(17);data=graph();labels=torch.arange(8)%2
    model=nil.graphs.hetero_sage({'paper':1,'author':2},8,{'author':2,'paper':2},[RELATION])
    optimizer=torch.optim.Adam(model.parameters(),lr=.03)
    # Labels on papers 4..7 are used only for held-out evaluation.
    for _ in range(40):
        optimizer.zero_grad();output=model(**data)
        loss=torch.nn.functional.cross_entropy(output['paper'][:4],labels[:4]);loss.backward();optimizer.step()
    with torch.no_grad():expected=model(**data)['paper'];accuracy=(expected[4:].argmax(1)==labels[4:]).float().mean().item()
    assert accuracy==1.
    snapshot=nil.graphs.save_hetero(**data,path=tmp_path/'graph')
    restored_data=nil.graphs.load_hetero(snapshot)
    nil.training.checkpoint.save(tmp_path/'model.pt',model)
    restored=nil.graphs.hetero_sage({'author':2,'paper':1},8,{'paper':2,'author':2},[RELATION])
    nil.training.checkpoint.load(tmp_path/'model.pt',restored)
    torch.testing.assert_close(expected,restored(**restored_data)['paper'],rtol=0,atol=0)
    renamed=nil.graphs.hetero_sage({'author':2,'paper':1},8,{'paper':2,'author':2},[('author','different','paper')])
    before=copy.deepcopy(renamed.state_dict())
    with pytest.raises(ValueError,match='structure'):nil.training.checkpoint.load(tmp_path/'model.pt',renamed)
    for key,value in before.items():torch.testing.assert_close(value,renamed.state_dict()[key],rtol=0,atol=0)


def test_schema_failure_empty_relation_and_double_graph_roundtrip(tmp_path):
    data=graph();data['features']={k:v.double() for k,v in data['features'].items()}
    data['edges'][RELATION]=torch.empty((2,0),dtype=torch.long)
    model=nil.graphs.hetero_sage({'author':2,'paper':1},4,{'author':2,'paper':2},[RELATION]).double()
    output=model(**data);assert torch.isfinite(output['paper']).all()
    restored=nil.graphs.load_hetero(nil.graphs.save_hetero(**data,path=tmp_path/'double'))
    assert restored['features']['paper'].dtype==torch.float64
    torch.testing.assert_close(model(**restored)['paper'],output['paper'],rtol=0,atol=0)
    with pytest.raises(ValueError,match='schema'):model(data['features'],{})
    data['edges'][RELATION]=torch.tensor([[0],[8]])
    with pytest.raises(ValueError,match='endpoints'):model(**data)


def test_projection_prefers_smaller_bipartite_intermediate():
    torch.manual_seed(1)
    features={'many':torch.randn(200,2),'few':torch.randn(2,1)}
    relation=('many','link','few');edges={relation:torch.tensor([list(range(200)),[i%2 for i in range(200)]])}
    model=nil.graphs.hetero_sage({'many':2,'few':1},4,{'many':2,'few':32},[relation])
    result=model(features,edges)
    assert result['few'].shape==(2,32)
    result['few'].square().mean().backward()
    assert model.first.messages[0].weight.grad is not None
