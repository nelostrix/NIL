import copy
import numpy as np
import pytest
torch=pytest.importorskip('torch'); torch.set_num_threads(1)
import nil


class RecurrentExport(torch.nn.Module):
    def __init__(self): super().__init__(); self.recurrent=torch.nn.LSTM(3,4,batch_first=True); self.head=torch.nn.Linear(4,2)
    def forward(self,x,h,c):
        output,(h,c)=self.recurrent(x,(h,c)); return self.head(output),h,c


def test_activation_checkpoint_gradient_rng_parity_reduced_saved_tensors_and_reload(tmp_path):
    torch.manual_seed(3)
    model=torch.nn.Sequential(torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
    wrapped=nil.optimization.activation_checkpoint(copy.deepcopy(model),segments=4)
    x=torch.randn(16,64,requires_grad=True)
    sizes=[]
    def measured(module):
        saved=[]
        with torch.autograd.graph.saved_tensors_hooks(lambda tensor:(saved.append(tensor.numel()*tensor.element_size()) or tensor),lambda tensor:tensor):
            loss=module(x).square().mean()
        sizes.append(sum(saved)); loss.backward()
    measured(model); measured(wrapped)
    assert sizes[1]<sizes[0]
    for left,right in zip(model.parameters(),wrapped.parameters()): torch.testing.assert_close(left.grad,right.grad,rtol=0,atol=0)
    data=nil.data.tensors(torch.randn(8,64),torch.randn(8,1))
    result=nil.train(wrapped,data,epochs=1,batch_size=4,device='cpu',output_dir=tmp_path)
    restored=nil.optimization.activation_checkpoint(copy.deepcopy(model),segments=4)
    from nil.training import checkpoint
    checkpoint.load(result.checkpoint,restored)
    torch.testing.assert_close(restored(x),wrapped(x),rtol=0,atol=0)


@pytest.mark.parametrize('kind',['cnn','lstm'])
def test_actual_onnx_cpu_runtime_architecture_parity(tmp_path,kind):
    pytest.importorskip('onnx'); pytest.importorskip('onnxruntime')
    if kind=='cnn':
        model=torch.nn.Sequential(torch.nn.Conv2d(1,2,3),torch.nn.ReLU(),torch.nn.Flatten(),torch.nn.Linear(8,1))
        example=torch.randn(2,1,4,4); names=['features']; output_names=['prediction']; axes={'features':{0:'batch'},'prediction':{0:'batch'}}
        probe=torch.randn(3,1,4,4); inputs=probe.numpy()
    else:
        model=RecurrentExport(); example=(torch.randn(2,4,3),torch.zeros(1,2,4),torch.zeros(1,2,4)); names=['features','hidden','cell']; output_names=['prediction','next_hidden','next_cell']
        axes={'features':{0:'batch',1:'time'},'hidden':{1:'batch'},'cell':{1:'batch'},'prediction':{0:'batch',1:'time'},'next_hidden':{1:'batch'},'next_cell':{1:'batch'}}
        probe=(torch.randn(3,5,3),torch.zeros(1,3,4),torch.zeros(1,3,4)); inputs={name:value.numpy() for name,value in zip(names,probe)}
    path=nil.optimization.export_onnx(model,example,tmp_path/f'{kind}.onnx',input_names=names,output_names=output_names,dynamic_axes=axes)
    runtime=nil.inference.OnnxModel(path); actual=runtime.predict(inputs)
    model.eval()
    with torch.inference_mode(): expected=model(*probe) if isinstance(probe,tuple) else model(probe)
    expected=expected if isinstance(expected,tuple) else (expected,)
    for result,reference in zip(actual,expected): np.testing.assert_allclose(result,reference.numpy(),atol=1e-5,rtol=1e-5)


def test_local_causal_model_heldout_perplexity_roundtrip(tmp_path):
    pytest.importorskip('transformers'); pytest.importorskip('peft')
    torch.manual_seed(7)
    texts=['robot moves safely','robot stops safely']*8
    tokenizer=nil.generative.tokenizer_from_texts(texts,vocab_size=270)
    train=nil.generative.text_dataset(texts,tokenizer,max_length=16)
    heldout=nil.generative.text_dataset(['robot moves','robot stops'],tokenizer,max_length=16)
    model=nil.generative.causal_lm(len(tokenizer),context_length=16,width=16,layers=1,heads=2)
    before=nil.generative.perplexity(model,heldout,device_name='cpu')
    trained=nil.train(model,train,task='causal_lm',epochs=3,batch_size=4,learning_rate=.01,device='cpu',output_dir=tmp_path/'runs')
    after=nil.generative.perplexity(trained.model,heldout,device_name='cpu')
    assert after['tokens']>0 and after['nll']<before['nll']
    destination=tmp_path/'model'; nil.generative.save(trained.model,tokenizer,destination)
    restored,_=nil.generative.load_causal_lm(destination)
    assert nil.generative.perplexity(restored,heldout,device_name='cpu')['nll']==pytest.approx(after['nll'],abs=1e-7)
