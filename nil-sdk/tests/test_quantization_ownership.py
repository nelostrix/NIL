"""Real dynamic quantization preserves caller mode, weights and module ownership."""
import copy
import pytest
torch=pytest.importorskip('torch');torch.set_num_threads(1)
import nil


@pytest.mark.parametrize('root',[False,True])
def test_quantization_owns_its_copy_and_preserves_the_caller(root):
    torch.manual_seed(7)
    original=torch.nn.Linear(4,3) if root else torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.ReLU(),torch.nn.Linear(8,3))
    original.train();before=copy.deepcopy(original.state_dict());x=torch.randn(8,4)
    with torch.no_grad():expected=original(x).clone()
    quantized=nil.optimization.quantize(original)
    assert original.training and quantized is not original and not quantized.training
    for name,value in original.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    with torch.no_grad():actual=quantized(x)
    torch.testing.assert_close(actual,expected,rtol=.05,atol=.02)
    with torch.no_grad():next(original.parameters()).fill_(0)
    torch.testing.assert_close(quantized(x),actual,rtol=0,atol=0)
    assert any('quantized.dynamic' in type(layer).__module__ for layer in quantized.modules())


def test_explicit_module_exclusion_and_per_channel_weights():
    model=torch.nn.ModuleDict({'body':torch.nn.Linear(4,4),'head':torch.nn.Linear(4,3)})
    quantized=nil.optimization.quantize(model,exclude=('head',),weight_scheme='per_channel')
    assert type(quantized['head']) is torch.nn.Linear and type(model['body']) is torch.nn.Linear
    assert 'quantized.dynamic' in type(quantized['body']).__module__
    assert quantized['body'].weight().qscheme() in (torch.per_channel_affine,torch.per_channel_symmetric)
    with pytest.raises(ValueError,match='Unknown'):nil.optimization.quantize(model,exclude=('missing',))
    with pytest.raises(ValueError,match='no supported'):nil.optimization.quantize(model,exclude=('',))
    with pytest.raises(ValueError,match='weight_scheme'):nil.optimization.quantize(model,weight_scheme='unknown')


def test_weight_only_storage_preserves_float_compute_fidelity_and_state_dict():
    torch.manual_seed(7)
    source=torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.ReLU(),torch.nn.Linear(32,4)).eval()
    x=torch.randn(8,16);expected=source(x)
    quantized=nil.optimization.quantize(source,method='weight_only_int8')
    assert quantized[0].qweight.dtype==torch.int8
    assert quantized[0].qweight.numel()==source[0].weight.numel()
    torch.testing.assert_close(quantized(x),expected,rtol=.03,atol=.01)
    restored=nil.optimization.quantize(source,method='weight_only_int8');restored.load_state_dict(quantized.state_dict())
    torch.testing.assert_close(restored(x),quantized(x),rtol=0,atol=0)
    assert type(source[0]) is torch.nn.Linear
    with pytest.raises(ValueError,match='float32'):quantized(x.double())
    with pytest.raises(ValueError,match='float32'):nil.optimization.quantize(torch.nn.Linear(2,2).double(),method='weight_only_int8')
