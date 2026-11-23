import math,os,subprocess,sys
import pytest
import nil.native as native


def test_broadcast_matmul_gradients_by_finite_differences():
    a=native.tensor([[.2,-.7],[1.2,.3]],requires_grad=True); b=native.tensor([[.4],[-.2]],requires_grad=True); bias=native.tensor([[.1]],requires_grad=True)
    loss=((a@b+bias).tanh()**2).mean(); loss.backward()
    def value(values): return ((native.tensor(values)@b+bias).tanh()**2).mean().item()
    for i in range(2):
        for j in range(2):
            plus=a.tolist(); minus=a.tolist(); plus[i][j]+=1e-6; minus[i][j]-=1e-6
            assert a.grad[i][j]==pytest.approx((value(plus)-value(minus))/2e-6,rel=1e-5,abs=1e-7)
    x=native.tensor([[2.,3.]],requires_grad=True); (x*x+x).sum().backward(); assert x.grad==((5.,7.),)


def test_regression_and_neural_imitation_converge_without_numpy(tmp_path):
    model=native.Linear(1,1,seed=7); x=[[i/10] for i in range(-10,11)]; y=[[2*r[0]+1] for r in x]
    result=native.fit(model,x,y,epochs=300,lr=.15); assert result['history'][-1]<1e-8
    classifier=native.Linear(1,1); result=native.fit(classifier,x,[[float(r[0]>0)] for r in x],epochs=200,lr=.5,loss='bce_logits'); assert result['history'][-1]<.25
    optimizer=native.SGD(model.parameters(),momentum=.8); optimizer.zero_grad(); native.mse(model(native.tensor(x)),native.tensor(y)).backward(); optimizer.step()
    path=native.save(model,tmp_path/'native.json',optimizer=optimizer); restored,other=native.load(path)
    assert restored.state_dict()==model.state_dict() and other.state_dict()==optimizer.state_dict()
    code="import nil.native as n,sys; m=n.Linear(1,1); assert n.fit(m,[[0],[1]],[[0],[1]],epochs=2)['steps']==2; assert not any(k in sys.modules for k in ('numpy','torch','sklearn'))"
    subprocess.run([sys.executable,'-S','-c',code],env={**os.environ,'PYTHONPATH':str(__import__('pathlib').Path(__file__).resolve().parents[1])},check=True)


def test_stale_graph_nonfinite_and_unsupported_device_rejected_before_mutation():
    x=native.tensor([[2.]],requires_grad=True); loss=(x*x).sum(); x.assign([[3.]])
    with pytest.raises(RuntimeError,match='changed'): loss.backward()
    assert x.grad is None
    for options in ({'dtype':'float32'},{'device':'cuda'}):
        with pytest.raises(ValueError): native.tensor([[1]],**options)
    for values in ([[float('nan')]],[[float('inf')]],[[True]],[[1],[2,3]]):
        with pytest.raises(ValueError): native.tensor(values)
    with pytest.raises(ValueError): native.tensor([[1,2]])@native.tensor([[1,2]])
    extreme=native.tensor([[1000.,-1000.]],requires_grad=True); loss=native.binary_cross_entropy(extreme,[[0,1]]); loss.backward(); assert math.isfinite(loss.item()) and extreme.grad==((.5,-.5),)


def test_native_multilayer_network_and_optimizer_gradient_parity(tmp_path):
    model=native.Sequential(native.Linear(2,4,seed=7),native.Tanh(),native.Linear(4,1,seed=8))
    x=[[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]]; y=[[0.],[1.],[1.],[0.]]
    result=native.fit(model,x,y,loss='bce_logits',epochs=1200,lr=.4)
    assert result['history'][-1]<.1
    loaded,_=native.load(native.save(model,tmp_path/'network.json'))
    assert loaded(native.tensor(x)).data==model(native.tensor(x)).data
    with pytest.raises(ValueError): native.Linear(100000,100000)


def test_native_gradient_matches_torch_double_precision_when_available():
    torch=pytest.importorskip('torch')
    left=native.tensor([[.2,-.4],[.7,.9]],requires_grad=True); right=native.tensor([[.1],[.8]],requires_grad=True)
    value=((left@right).sigmoid().log()).mean(); value.backward()
    a=torch.tensor(left.tolist(),dtype=torch.float64,requires_grad=True); b=torch.tensor(right.tolist(),dtype=torch.float64,requires_grad=True)
    torch.log(torch.sigmoid(a@b)).mean().backward()
    for i in range(2):
        for j in range(2): assert left.grad[i][j]==pytest.approx(a.grad[i,j].item(),abs=1e-12)

@pytest.mark.parametrize('operation',[
    lambda x:((x.exp()+2).log()/3).sum(),
    lambda x:((x.sigmoid()**2)*x).mean(),
    lambda x:(x.relu()+1/x).sum(),
    lambda x:native.binary_cross_entropy(x,native.Tensor([[0.,1.],[1.,0.]])),
])
def test_operator_gradients_independently_by_central_difference(operation):
    data=[[.3,-.8],[1.1,-.4]]; x=native.Tensor(data,requires_grad=True)
    operation(x).backward()
    for i in range(2):
        for j in range(2):
            plus=[row[:] for row in data]; minus=[row[:] for row in data]
            plus[i][j]+=1e-6; minus[i][j]-=1e-6
            expected=(operation(native.Tensor(plus)).item()-operation(native.Tensor(minus)).item())/2e-6
            assert x.grad[i][j]==pytest.approx(expected,rel=1e-5,abs=1e-7)


def test_matmul_rejects_oversized_output_before_materialization():
    with pytest.raises(ValueError,match='output exceeds'):
        native.Tensor([[1.]]*200) @ native.Tensor([[1.]*1000])
