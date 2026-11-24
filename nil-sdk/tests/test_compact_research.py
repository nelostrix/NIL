"""Framework compatibility, not compact-intelligence capability acceptance."""
import copy
from pathlib import Path
import runpy
import pytest

torch=pytest.importorskip('torch')
import nil


@pytest.fixture(scope='module')
def recipe():
    return runpy.run_path(str(Path(__file__).resolve().parents[1]/'examples/compact_research.py'))


def test_shared_depth_does_not_add_parameters_and_gradients_accumulate(recipe):
    torch.set_num_threads(1);torch.manual_seed(3)
    build=recipe['build_model']
    shared=build(depth=2);deeper=build(depth=6)
    untied=build(shared=False,depth=2)
    assert sum(p.numel() for p in shared.parameters())==sum(p.numel() for p in deeper.parameters())
    assert sum(p.numel() for p in untied.parameters())>sum(p.numel() for p in shared.parameters())
    untied.embedding.load_state_dict(shared.embedding.state_dict())
    untied.head.load_state_dict(shared.head.state_dict())
    with torch.no_grad():untied.position.copy_(shared.position)
    for block in untied.blocks:block.load_state_dict(shared.blocks[0].state_dict())
    x=torch.randint(0,16,(4,6));y=x[:,0]%2
    torch.testing.assert_close(shared(x),untied(x),rtol=0,atol=0)
    torch.nn.functional.cross_entropy(shared(x),y).backward()
    torch.nn.functional.cross_entropy(untied(x),y).backward()
    gradients=[dict(block.named_parameters()) for block in untied.blocks]
    for name,p in shared.blocks[0].named_parameters():
        expected=sum(group[name].grad for group in gradients)
        torch.testing.assert_close(p.grad,expected,rtol=1e-5,atol=1e-6)


def test_distillation_matches_explicit_kl_and_freezes_teacher(recipe):
    task=recipe['DistillationStep'](alpha=.3,temperature=2.)
    student=torch.nn.Linear(3,2)
    x=torch.tensor([[1.,2.,0.],[-1.,0.,3.]])
    y=torch.tensor([0,1]);teacher=torch.tensor([[2.,-1.],[.5,1.]],requires_grad=True)
    loss=task(student,(x,y,teacher))
    logits=student(x);p=(teacher.detach()/2).softmax(-1)
    expected=.3*torch.nn.functional.cross_entropy(logits,y)+.7*4*(p*(p.log()-(logits/2).log_softmax(-1))).sum()/len(y)
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert teacher.grad is None and student.weight.grad is not None
    with pytest.raises(ValueError,match='finite'):task(student,(x,y,teacher.detach()*float('nan')))
    with pytest.raises(ValueError,match='aligned'):task(student,(x,y,torch.ones(2,3)))


def test_distillation_uneven_accumulation_matches_one_full_batch(recipe,tmp_path):
    torch.manual_seed(4)
    model=torch.nn.Linear(3,2);expected=copy.deepcopy(model)
    x=torch.randn(5,3);y=torch.tensor([0,1,0,1,1]);logits=torch.randn(5,2)
    task=recipe['DistillationStep']()
    task(expected,(x,y,logits)).backward()
    torch.optim.SGD(expected.parameters(),lr=.01).step()
    nil.Trainer(model,task=task,optimizer=torch.optim.SGD(model.parameters(),lr=.01),
        config=nil.TrainingConfig(epochs=1,batch_size=2,accumulation_steps=3,
            shuffle=False,device='cpu',output_dir=tmp_path)).fit(nil.data.tensors(x,y,logits))
    for actual,reference in zip(model.parameters(),expected.parameters()):
        torch.testing.assert_close(actual,reference,rtol=1e-6,atol=1e-6)


def test_reused_depth_checkpoint_guard_and_exact_resume(recipe,tmp_path):
    torch.set_num_threads(1);torch.manual_seed(9)
    initial=recipe['build_model'](depth=2)
    x=torch.randint(0,16,(16,6));y=x[:,0]%2;data=nil.data.tensors(x,y)
    options=dict(task='classification',batch_size=8,learning_rate=.01,device='cpu',seed=7)
    full=nil.train(copy.deepcopy(initial),data,epochs=4,output_dir=tmp_path/'full',**options)
    first=nil.train(copy.deepcopy(initial),data,epochs=2,output_dir=tmp_path/'first',**options)
    resumed=nil.train(copy.deepcopy(initial),data,epochs=4,resume=first.checkpoint,output_dir=tmp_path/'resumed',**options)
    assert full.history==resumed.history
    for key,value in full.model.state_dict().items():
        torch.testing.assert_close(value,resumed.model.state_dict()[key],rtol=0,atol=0)
    changed=recipe['build_model'](depth=3);before=copy.deepcopy(changed.state_dict())
    with pytest.raises(ValueError,match='structure'):
        nil.training.checkpoint.load(first.checkpoint,changed)
    for key,value in before.items():torch.testing.assert_close(value,changed.state_dict()[key],rtol=0,atol=0)


@pytest.mark.parametrize('options',[{'depth':0},{'shared':1},{'width':7,'heads':2}])
def test_invalid_architecture(recipe,options):
    with pytest.raises(ValueError):recipe['build_model'](**options)


@pytest.mark.parametrize('options',[{'alpha':True},{'temperature':float('inf')},{'alpha':1.1}])
def test_invalid_distillation(recipe,options):
    with pytest.raises(ValueError):recipe['DistillationStep'](**options)
