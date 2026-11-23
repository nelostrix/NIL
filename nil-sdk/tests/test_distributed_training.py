import copy
from pathlib import Path
import pytest

torch=pytest.importorskip('torch'); torch.set_num_threads(1)
from nil import distributed


def record_callback(model,row):
    marker=Path(__import__('os').environ['NIL_DDP_TEST_MARKER'])
    with marker.open('a') as stream: stream.write(str(row['epoch'])+'\n')


def fail_worker_callback(model,row):
    if row['epoch']==2: raise RuntimeError('deliberate rank-zero failure')


def test_cpu_ddp_matches_global_batches_with_uneven_shards_and_resume(tmp_path,monkeypatch):
    torch.manual_seed(7); initial=torch.nn.Linear(2,1)
    x=torch.arange(22).float().reshape(11,2)/10; y=x.sum(dim=1,keepdim=True)
    reference=copy.deepcopy(initial); optimizer=torch.optim.SGD(reference.parameters(),lr=.03,momentum=.8)
    # Batches of 4,4,3 partition into unequal per-rank sizes without duplication.
    for epoch in range(3):
        order=torch.randperm(len(x),generator=torch.Generator().manual_seed(3+epoch))
        for start in range(0,len(x),4):
            indices=order[start:start+4]; optimizer.zero_grad(); torch.nn.functional.mse_loss(reference(x[indices]),y[indices]).backward(); optimizer.step()
    marker=tmp_path/'callbacks'; monkeypatch.setenv('NIL_DDP_TEST_MARKER',str(marker))
    result=distributed.train(copy.deepcopy(initial),x,y,epochs=3,batch_size=4,world_size=2,lr=.03,momentum=.8,seed=3,output_dir=tmp_path/'full',callback=record_callback)
    assert marker.read_text().splitlines()==['1','2','3']
    for left,right in zip(result.model.parameters(),reference.parameters()): torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)
    first=distributed.train(copy.deepcopy(initial),x,y,epochs=1,batch_size=4,world_size=2,lr=.03,momentum=.8,seed=3,output_dir=tmp_path/'partial')
    resumed=distributed.train(copy.deepcopy(initial),x,y,epochs=3,batch_size=4,world_size=2,lr=.03,momentum=.8,seed=3,output_dir=tmp_path/'resumed',resume=first.checkpoint)
    assert result.history==resumed.history
    for left,right in zip(result.model.parameters(),resumed.model.parameters()): torch.testing.assert_close(left,right,rtol=0,atol=0)


def test_rank_failure_keeps_previous_checkpoint_and_caller_unchanged(tmp_path):
    model=torch.nn.Linear(1,1); before=copy.deepcopy(model.state_dict()); x=torch.ones(8,1); y=x*2
    with pytest.raises(Exception,match='deliberate rank-zero failure'):
        distributed.train(model,x,y,epochs=3,batch_size=4,output_dir=tmp_path/'failed',callback=fail_worker_callback,timeout=40)
    from nil.training import checkpoint
    assert checkpoint.read(tmp_path/'failed/latest.pt')['state']['epoch']==1
    for name,value in model.state_dict().items(): torch.testing.assert_close(value,before[name],rtol=0,atol=0)


def test_bounded_inputs_reject_before_spawning(tmp_path):
    model=torch.nn.Linear(2,1);x=torch.ones(4,2);y=torch.ones(4,1)
    for options,pattern in [({'max_parameters':1},'max_parameters'),({'max_dataset_bytes':1},'max_dataset_bytes'),({'seed':True},'seed')]:
        with pytest.raises(ValueError,match=pattern):distributed.train(model,x,y,output_dir=tmp_path/'not-created',**options)
    with pytest.raises(ValueError,match='Finite matching'):
        distributed.train(model,x.to(torch.complex64),y,output_dir=tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()


def hang_callback(model,row):
    if row['epoch']==2:
        import os,time
        Path(os.environ['NIL_DDP_HANG_MARKER']).write_text(str(os.getpid()))
        time.sleep(30)


def test_hung_rank_deadline_terminates_children_and_retains_epoch(tmp_path,monkeypatch):
    import os,time
    marker=tmp_path/'hung';monkeypatch.setenv('NIL_DDP_HANG_MARKER',str(marker))
    model=torch.nn.Linear(1,1);before=copy.deepcopy(model.state_dict());started=time.monotonic()
    with pytest.raises(Exception):distributed.train(model,torch.ones(8,1),torch.ones(8,1),epochs=3,batch_size=4,output_dir=tmp_path/'run',callback=hang_callback,timeout=6)
    assert marker.exists() and time.monotonic()-started<15
    from nil.execution import _identity
    if __import__('sys').platform=='linux':assert _identity(int(marker.read_text())) is None
    from nil.training import checkpoint
    assert checkpoint.read(tmp_path/'run/latest.pt')['state']['epoch']==1
    for name,value in model.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)
