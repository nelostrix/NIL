"""Actual GPU gates. CPU skips are rejected by scripts/accept_gpu.py."""
import copy,os,time
import pytest
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
torch=pytest.importorskip('torch')
import nil
from nil.training import checkpoint
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires actual CUDA hardware')


def _data():
    generator=torch.Generator().manual_seed(3)
    x=torch.randn(29,3,generator=generator); return nil.data.tensors(x,x.sum(1,keepdim=True))


@pytest.mark.parametrize('precision',['fp16','bf16'])
def test_cuda_learning_accumulation_clipping_and_resume(tmp_path,record_property,precision):
    if precision=='bf16' and not torch.cuda.is_bf16_supported(): pytest.skip('GPU lacks native BF16 support')
    torch.manual_seed(3); initial=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.Tanh(),torch.nn.Linear(8,1))
    options=dict(precision=precision,device='cuda',batch_size=5,accumulation_steps=2,clip_grad_norm=1.,learning_rate=.03,seed=7)
    torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    full=nil.train(copy.deepcopy(initial),_data(),epochs=6,output_dir=tmp_path/'full',**options)
    first=nil.train(copy.deepcopy(initial),_data(),epochs=2,output_dir=tmp_path/'first',**options)
    resumed=nil.train(copy.deepcopy(initial),_data(),epochs=6,resume=first.checkpoint,output_dir=tmp_path/'resume',**options)
    assert full.history[-1]['loss']<full.history[0]['loss']
    assert full.history==resumed.history
    for left,right in zip(full.model.parameters(),resumed.model.parameters()):torch.testing.assert_close(left,right,rtol=0,atol=0)
    assert checkpoint.read(full.checkpoint)['state']['amp_scaler']==checkpoint.read(resumed.checkpoint)['state']['amp_scaler']
    torch.cuda.synchronize();record_property('seconds',time.perf_counter()-started)
    record_property('peak_allocated_bytes',torch.cuda.max_memory_allocated());record_property('precision',precision)


def test_cuda_fp16_overflow_skips_update_and_unscales_before_clip(tmp_path,record_property):
    # Exercise NIL's update path with native GradScaler, not a simulated scaler.
    model=torch.nn.Linear(1,1); trainer=nil.Trainer(model,config=nil.TrainingConfig(device='cuda',precision='fp16',clip_grad_norm=.1,output_dir=tmp_path))
    trainer.optimizer.zero_grad(set_to_none=True); initial=[p.detach().clone() for p in trainer.model.parameters()]
    loss=trainer.model(torch.ones(2,1,device='cuda')).square().mean()
    trainer.scaler.scale(loss).backward(); before=trainer.scaler.get_scale()
    next(trainer.model.parameters()).grad.fill_(float('inf'))
    assert trainer._update(2.) is False and trainer._amp_skipped==1
    assert trainer.scaler.get_scale()<before
    for parameter,expected in zip(trainer.model.parameters(),initial):torch.testing.assert_close(parameter,expected,rtol=0,atol=0)
    # Clipping sees finite unscaled gradients, bounded by the configured norm.
    loss=trainer.model(torch.ones(2,1,device='cuda')).square().mean();trainer.scaler.scale(loss).backward()
    observed=[]; original=trainer.torch.nn.utils.clip_grad_norm_
    def clip(parameters,*args,**kwargs):
        parameters=list(parameters); observed.append([p.grad.clone() for p in parameters]);return original(parameters,*args,**kwargs)
    from unittest.mock import patch
    expected=[p.grad.detach().clone()/trainer.scaler.get_scale()/2 for p in trainer.model.parameters()]
    with patch.object(trainer.torch.nn.utils,'clip_grad_norm_',clip):assert trainer._update(2.) is True
    for actual,wanted in zip(observed[0],expected):torch.testing.assert_close(actual,wanted,rtol=1e-6,atol=1e-7)
    record_property('scale_before_overflow',before);record_property('scale_after_overflow',trainer.scaler.get_scale())


def fail_rank(model,row):
    if row['epoch']==2:raise RuntimeError('GPU rank fault')


@pytest.mark.parametrize('strategy',['ddp','fsdp'])
def test_cuda_distributed_serial_parity_resume_and_failure(tmp_path,record_property,strategy):
    if torch.cuda.device_count()<2 or not torch.distributed.is_nccl_available():pytest.skip('Requires two CUDA devices and NCCL')
    torch.manual_seed(3); initial=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Tanh(),torch.nn.Linear(4,1))
    x=torch.arange(22).float().reshape(11,2)/10;y=x.sum(1,keepdim=True)
    options=dict(strategy=strategy,device='cuda',world_size=2,batch_size=4,seed=7,lr=.03,momentum=.8,timeout=90)
    full=nil.distributed.train(copy.deepcopy(initial),x,y,epochs=3,output_dir=tmp_path/'full',**options)
    first=nil.distributed.train(copy.deepcopy(initial),x,y,epochs=1,output_dir=tmp_path/'first',**options)
    resumed=nil.distributed.train(copy.deepcopy(initial),x,y,epochs=3,resume=first.checkpoint,trusted_resume=True,output_dir=tmp_path/'resume',**options)
    for left,right in zip(full.model.parameters(),resumed.model.parameters()):torch.testing.assert_close(left,right,rtol=0,atol=0)
    serial=copy.deepcopy(initial).cuda();optimizer=torch.optim.SGD(serial.parameters(),lr=.03,momentum=.8)
    for epoch in range(3):
        order=torch.randperm(11,generator=torch.Generator().manual_seed(7+epoch))
        for offset in range(0,11,4):
            indices=order[offset:offset+4];optimizer.zero_grad();torch.nn.functional.mse_loss(serial(x[indices].cuda()),y[indices].cuda()).backward();optimizer.step()
    for left,right in zip(full.model.parameters(),serial.parameters()):torch.testing.assert_close(left,right.cpu(),rtol=2e-4,atol=1e-5)
    caller=copy.deepcopy(initial); saved=copy.deepcopy(caller.state_dict())
    with pytest.raises(Exception,match='GPU rank fault'):
        nil.distributed.train(caller,x,y,epochs=3,callback=fail_rank,output_dir=tmp_path/'fault',**options)
    for name,value in caller.state_dict().items():torch.testing.assert_close(value,saved[name],rtol=0,atol=0)
    path=tmp_path/'fault'/('shards/latest.json' if strategy=='fsdp' else 'latest.pt')
    recovered=nil.distributed.train(caller,x,y,epochs=2,resume=path,trusted_resume=True,output_dir=tmp_path/'recover',**options)
    assert len(recovered.history)==2
    assert all(row['peak_allocated_bytes']>0 for row in full.diagnostics['ranks'])
    record_property('worker_memory',full.diagnostics)
