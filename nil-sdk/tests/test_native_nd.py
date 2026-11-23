"""Shape semantics, independent derivatives, Torch parity and ND optimizer resume."""
import copy
import math
import random
import pytest
from nil import native as n
from nil._native_tensor import unpack


def values(shape):
    rng=random.Random(7)
    return [rng.uniform(-.8,.8) for _ in range(math.prod(shape))]


def test_shapes_contiguous_strides_scalar_migration_and_copy_safety():
    assert n.Tensor(2).shape==(1,1) and n.Tensor([1,2]).shape==(1,2)
    assert n.Tensor(2,shape=()).shape==()
    vector=n.Tensor([1,2],shape=(2,));assert vector.shape==(2,)
    x=n.Tensor(list(range(24)),shape=(2,3,4),requires_grad=True)
    assert x.strides==(12,4,1) and x.numel()==24 and x.ndim==3
    assert x[1,-1,2].shape==() and x[1,-1,2].item()==22
    assert x[...,1].shape==(2,3)
    assert x[:,::-1,::2].shape==(2,3,2)
    assert x.reshape(4,-1).shape==(4,6)
    assert x.permute(2,0,1).shape==(4,2,3)
    assert x.transpose(0,-1).shape==(4,3,2)
    copied=n.Tensor([1,2],shape=(2,));other=copied.reshape(1,2);other.assign([[4,5]])
    assert copied.tolist()==[1.,2.]
    with pytest.raises(RuntimeError,match='leaf'):x.reshape(24).assign(list(range(24)))


@pytest.mark.parametrize('shape',[(0,),(-2,),(True,),(100001,),tuple([1]*9)])
def test_invalid_shapes_are_rejected(shape):
    with pytest.raises(ValueError):n.Tensor([1],shape=shape)


def test_invalid_axes_empty_slices_shape_assign_and_stale_graph_refuse():
    x=n.Tensor(values((2,3,4)),shape=(2,3,4),requires_grad=True)
    for operation in (lambda:x.sum(axis=(0,0)),lambda:x.mean(axis=True),lambda:x.permute(0,0,1),
                      lambda:x.reshape(-1,-1),lambda:x.reshape(5,5),lambda:x[:,0:0],lambda:x[True]):
        with pytest.raises((ValueError,TypeError)):operation()
    assert x.sum(axis=()).shape==x.shape
    assert x.sum(axis=(0,2),keepdims=True).shape==(1,3,1)
    loss=x.reshape(4,6).sum();x.assign(x.tolist())
    with pytest.raises(RuntimeError,match='changed'):loss.backward()
    assert x.grad is None


@pytest.mark.parametrize('case',['broadcast','slice','reduce','softmax','matmul'])
def test_nd_gradients_match_independent_central_differences(case):
    shape=(2,2,3);data=values(shape)
    def operation(x):
        if case=='broadcast':return ((x*n.Tensor([.2,.7,-.3],shape=(3,))-.4).tanh()**2).mean()
        if case=='slice':return (x.permute(2,0,1)[:,1,::-1].reshape(3,2)**2).sum()
        if case=='reduce':return (x.mean(axis=(0,2),keepdims=True)**2).sum()
        if case=='softmax':return (x.log_softmax(axis=1)**2).mean()
        return ((x@n.Tensor(values((1,3,2)),shape=(1,3,2)))**2).mean()
    x=n.Tensor(data,shape=shape,requires_grad=True);operation(x).backward();gradient=unpack(x.grad)[1]
    for i in range(len(data)):
        plus=data[:];minus=data[:];plus[i]+=1e-6;minus[i]-=1e-6
        expected=(operation(n.Tensor(plus,shape=shape)).item()-operation(n.Tensor(minus,shape=shape)).item())/2e-6
        assert gradient[i]==pytest.approx(expected,rel=1e-5,abs=1e-7)


@pytest.mark.parametrize('left,right',[( (2,1,3,4),(1,5,4,2)),((4,),(4,)),((4,),(2,4,3)),((2,3,4),(4,))])
def test_broadcast_batched_and_vector_matmul_match_torch_values_and_gradients(left,right):
    torch=pytest.importorskip('torch')
    a=n.Tensor(values(left),shape=left,requires_grad=True);b=n.Tensor(values(right),shape=right,requires_grad=True)
    x=torch.tensor(a.tolist(),dtype=torch.float64,requires_grad=True);y=torch.tensor(b.tolist(),dtype=torch.float64,requires_grad=True)
    result=a@b;reference=x@y
    assert result.shape==tuple(reference.shape)
    torch.testing.assert_close(torch.tensor(result.tolist(),dtype=torch.float64),reference)
    (result**2).mean().backward();reference.square().mean().backward()
    torch.testing.assert_close(torch.tensor(a.grad,dtype=torch.float64),x.grad,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(torch.tensor(b.grad,dtype=torch.float64),y.grad,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('optimizer',['SGD','Adam'])
@pytest.mark.parametrize('shape',[(),(3,),(2,2,2)])
def test_nd_optimizer_exact_state_continuation_and_torch_parity(optimizer,shape):
    torch=pytest.importorskip('torch');data=values(shape)
    p=n.Tensor(data,shape=shape,requires_grad=True)
    reference=torch.tensor(p.tolist(),dtype=torch.float64,requires_grad=True)
    options={'lr':.03,'weight_decay':.01}
    if optimizer=='SGD':options['momentum']=.8
    ours=getattr(n,optimizer)([p],**options);theirs=getattr(torch.optim,optimizer)([reference],**options,foreach=False)
    for _ in range(4):
        ours.zero_grad();theirs.zero_grad();(p**2).mean().backward();reference.square().mean().backward();ours.step();theirs.step()
    torch.testing.assert_close(torch.tensor(p.tolist(),dtype=torch.float64),reference,rtol=1e-12,atol=1e-12)
    restored=n.Tensor(p.tolist(),shape=shape,requires_grad=True);other=getattr(n,optimizer)([restored],**options)
    other.load_state_dict(copy.deepcopy(ours.state_dict()))
    for _ in range(4):
        for parameter,opt in ((p,ours),(restored,other)):
            opt.zero_grad();(parameter**2).mean().backward();opt.step()
    assert p.tolist()==restored.tolist() and ours.state_dict()==other.state_dict()


def test_nd_losses_and_tensor_persistence_match_torch(tmp_path):
    torch=pytest.importorskip('torch')
    x=n.Tensor(values((2,3,4)),shape=(2,3,4),requires_grad=True)
    labels=[[0,1,2],[3,2,1]]
    value=n.cross_entropy(x,labels);value.backward()
    reference=torch.tensor(x.tolist(),dtype=torch.float64,requires_grad=True)
    loss=torch.nn.functional.cross_entropy(reference.reshape(-1,4),torch.tensor(labels).reshape(-1));loss.backward()
    assert value.item()==pytest.approx(loss.item(),abs=1e-12)
    torch.testing.assert_close(torch.tensor(x.grad,dtype=torch.float64),reference.grad,rtol=1e-12,atol=1e-12)
    restored=n.load_tensor(n.save_tensor(x,tmp_path/'tensor.json'))
    assert restored.shape==x.shape and restored.tolist()==x.tolist() and restored.requires_grad
    assert restored.grad is None
    x.zero_grad();targets=[[[0.,1.,0.,1.]]*3]*2
    result=n.binary_cross_entropy(x,targets);result.backward()
    reference.grad=None;expected=torch.nn.functional.binary_cross_entropy_with_logits(reference,torch.tensor(targets,dtype=torch.float64));expected.backward()
    assert result.item()==pytest.approx(expected.item(),abs=1e-12)
    torch.testing.assert_close(torch.tensor(x.grad,dtype=torch.float64),reference.grad,rtol=1e-12,atol=1e-12)
    with pytest.raises(ValueError):n.binary_cross_entropy(n.Tensor([[1,2],[3,4]]),[[0,1,0,1]])
