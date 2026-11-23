"""Independent gradients, Torch parity, real learning and exact Adam continuation."""
import copy
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import nil.native as n


@pytest.mark.parametrize('operation',[
    lambda x:(x.softmax()*n.Tensor([[.2,-.7,.4],[.3,.8,-.2]])).sum(),
    lambda x:(x.log_softmax()**2).mean(),
    lambda x:n.cross_entropy(x,[2,0]),
])
def test_multiclass_gradients_match_independent_finite_differences(operation):
    data=[[.3,-.8,1.2],[1.1,-.4,.1]];x=n.Tensor(data,requires_grad=True)
    operation(x).backward()
    for i,row in enumerate(data):
        for j,_ in enumerate(row):
            plus=copy.deepcopy(data);minus=copy.deepcopy(data)
            plus[i][j]+=1e-6;minus[i][j]-=1e-6
            expected=(operation(n.Tensor(plus)).item()-operation(n.Tensor(minus)).item())/2e-6
            assert x.grad[i][j]==pytest.approx(expected,rel=1e-5,abs=1e-7)


def test_extreme_logits_shift_invariance_and_label_guards():
    x=n.Tensor([[1000,-1000,0],[-1000,1000,0]],requires_grad=True)
    loss=n.cross_entropy(x,[1,2]);loss.backward()
    assert loss.item()==pytest.approx(1500.)
    assert all(math.isfinite(value) for row in x.grad for value in row)
    assert x.softmax().tolist()==[[1.,0.,0.],[0.,1.,0.]]
    assert n.cross_entropy(x+10000,[1,2]).item()==loss.item()
    for targets in ([True,0],[-1,0],[3,0],[1.0,0],[1],[[1],[0]]):
        with pytest.raises(ValueError):n.cross_entropy(x,targets)
    stale=x.log_softmax().sum();x.assign([[0,0,0],[0,0,0]])
    with pytest.raises(RuntimeError,match='changed'):stale.backward()


def test_adam_exact_checkpoint_continuation_and_multiclass_learning(tmp_path):
    features=n.Tensor([[1,0],[0,1],[-1,-1]])
    model=n.Linear(2,3,seed=7);optimizer=n.Adam(model.parameters(),lr=.04,betas=(.8,.95),weight_decay=.001)
    def step(model,optimizer):
        optimizer.zero_grad();loss=n.cross_entropy(model(features),[0,1,2]);loss.backward();optimizer.step()
        return loss.item()
    before=step(model,optimizer)
    for _ in range(19):step(model,optimizer)
    path=n.save(model,tmp_path/'adam.json',optimizer=optimizer)
    restored,other=n.load(path)
    assert isinstance(other,n.Adam)
    assert other.state_dict()==optimizer.state_dict()
    for _ in range(30):assert step(model,optimizer)==step(restored,other)
    assert model.state_dict()==restored.state_dict() and optimizer.state_dict()==other.state_dict()
    assert n.cross_entropy(model(features),[0,1,2]).item()<before/5
    result=n.fit(n.Linear(2,3),features.tolist(),[0,1,2],epochs=100,lr=.05,optimizer='adam',loss='cross_entropy')
    assert result['history'][-1]<.1
    with pytest.raises(ValueError,match='match the saved model'):
        n.save(model,tmp_path/'wrong-optimizer.json',optimizer=n.Adam(n.Linear(2,3).parameters()))
    assert not (tmp_path/'wrong-optimizer.json').exists()


def test_adam_rejects_bad_gradients_state_and_options_transactionally():
    model=n.Linear(1,1);optimizer=n.Adam(model.parameters());parameters=model.parameters()
    for parameter in parameters:parameter.grad=((.5,),)
    optimizer.step();model_before=copy.deepcopy(model.state_dict());before=optimizer.state_dict()
    for invalid in (None,((float('nan'),),),((1,2),)):
        parameters[-1].grad=invalid
        with pytest.raises(ValueError):optimizer.step()
        assert model.state_dict()==model_before and optimizer.state_dict()==before
    for key,value in (('steps',True),('betas',[.8,.99]),('second',[[[-1]],[[0]]]),('first',[[[1,2]],[[0]]])):
        corrupt=copy.deepcopy(before);corrupt[key]=value
        with pytest.raises(ValueError):optimizer.load_state_dict(corrupt)
        assert optimizer.state_dict()==before
    detached=optimizer.state_dict();detached['first'][0][0][0]=999
    assert optimizer.state_dict()==before
    for options in ({'lr':True},{'eps':0},{'betas':(.9,1.)},{'weight_decay':float('inf')}):
        with pytest.raises(ValueError):n.Adam(parameters,**options)


def test_softmax_and_adam_match_torch_float64():
    torch=pytest.importorskip('torch')
    values=[[.3,-.8,1.2],[1.1,-.4,.1]]
    x=n.Tensor(values,requires_grad=True);reference=torch.tensor(values,dtype=torch.float64,requires_grad=True)
    optimizer=n.Adam([x],lr=.03,betas=(.8,.95),eps=1e-7,weight_decay=.02)
    other=torch.optim.Adam([reference],lr=.03,betas=(.8,.95),eps=1e-7,weight_decay=.02,foreach=False)
    for _ in range(8):
        optimizer.zero_grad();other.zero_grad()
        loss=n.cross_entropy(x,[2,0]);expected=torch.nn.functional.cross_entropy(reference,torch.tensor([2,0]))
        assert loss.item()==pytest.approx(expected.item(),abs=1e-12)
        loss.backward();expected.backward()
        torch.testing.assert_close(torch.tensor(x.grad,dtype=torch.float64),reference.grad,rtol=1e-12,atol=1e-12)
        optimizer.step();other.step()
        torch.testing.assert_close(torch.tensor(x.tolist(),dtype=torch.float64),reference,rtol=1e-12,atol=1e-12)


def test_new_native_workflow_needs_no_site_packages():
    code="import nil.native as n,sys; m=n.Linear(2,3); r=n.fit(m,[[1,0],[0,1],[-1,-1]],[0,1,2],epochs=5,optimizer='adam',loss='cross_entropy'); assert r['steps']==5; assert not any(k in sys.modules for k in ('torch','numpy','sklearn'))"
    subprocess.run([sys.executable,'-S','-c',code],env={**os.environ,'PYTHONPATH':str(Path(n.__file__).resolve().parent.parent)},check=True,timeout=10)
