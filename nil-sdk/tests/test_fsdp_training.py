import copy,json
from pathlib import Path
import pytest
torch=pytest.importorskip('torch');torch.set_num_threads(1)
from nil import distributed


def fail_callback(model,row):
    if row['epoch']==2: raise RuntimeError('sharded rank-zero fault')


def test_actual_cpu_shards_match_serial_and_exact_resume(tmp_path):
    torch.manual_seed(3); initial=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Tanh(),torch.nn.Linear(4,1))
    x=torch.arange(22).float().reshape(11,2)/10; y=x.sum(1,keepdim=True)
    options=dict(strategy='fsdp',world_size=2,batch_size=4,lr=.03,momentum=.8,seed=7,timeout=60)
    full=distributed.train(copy.deepcopy(initial),x,y,epochs=3,output_dir=tmp_path/'full',**options)
    serial=copy.deepcopy(initial); optimizer=torch.optim.SGD(serial.parameters(),lr=.03,momentum=.8)
    for epoch in range(3):
        order=torch.randperm(11,generator=torch.Generator().manual_seed(7+epoch))
        for offset in range(0,11,4):
            indices=order[offset:offset+4];optimizer.zero_grad();torch.nn.functional.mse_loss(serial(x[indices]),y[indices]).backward();optimizer.step()
    for left,right in zip(serial.parameters(),full.model.parameters()):torch.testing.assert_close(left,right,rtol=2e-5,atol=1e-6)
    manifest=json.loads(full.checkpoint.read_text()); assert len(manifest['files'])==2
    count=sum(p.numel() for p in initial.parameters())
    assert all(row['parameter_elements']<count for row in full.diagnostics['ranks'])
    assert len(list(full.checkpoint.parent.glob('epoch-*')))==2
    first=distributed.train(copy.deepcopy(initial),x,y,epochs=1,output_dir=tmp_path/'partial',**options)
    with pytest.raises(ValueError,match='trusted_resume=True'):
        distributed.train(copy.deepcopy(initial),x,y,epochs=3,resume=first.checkpoint,output_dir=tmp_path/'untrusted',**options)
    resumed=distributed.train(copy.deepcopy(initial),x,y,epochs=3,resume=first.checkpoint,trusted_resume=True,output_dir=tmp_path/'resumed',**options)
    export_only=distributed.train(copy.deepcopy(initial),x,y,epochs=1,resume=first.checkpoint,trusted_resume=True,output_dir=tmp_path/'export-only',**options)
    assert export_only.history==first.history
    for left,right in zip(export_only.model.parameters(),first.model.parameters()):torch.testing.assert_close(left,right,rtol=0,atol=0)
    assert export_only.export_checkpoint.is_file() and export_only.checkpoint==first.checkpoint.resolve()
    assert full.history==resumed.history
    for left,right in zip(full.model.parameters(),resumed.model.parameters()):torch.testing.assert_close(left,right,rtol=0,atol=0)
    with pytest.raises(ValueError,match='identical'):
        distributed.train(copy.deepcopy(initial),x,y+1,epochs=4,resume=first.checkpoint,trusted_resume=True,output_dir=tmp_path/'mismatch',**options)
    with pytest.raises(ValueError,match='identical'):
        distributed.train(copy.deepcopy(initial),x,y,epochs=4,resume=first.checkpoint,trusted_resume=True,output_dir=tmp_path/'world-mismatch',**{**options,'world_size':3})
    (first.checkpoint.parent/json.loads(first.checkpoint.read_text())['directory']/'rank-1.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='integrity'):
        distributed.train(copy.deepcopy(initial),x,y,epochs=4,resume=first.checkpoint,trusted_resume=True,output_dir=tmp_path/'tamper',**options)


def fail_publication(model,row):
    if row['epoch']==2:
        import nil._sharded as shards
        def fault(*args,**kwargs):raise OSError('shard publication fault')
        shards.atomic_json=fault


@pytest.mark.parametrize('callback,message',[(fail_callback,'sharded rank-zero fault'),(fail_publication,'shard publication fault')])
def test_rank_failure_retains_committed_shards_and_caller(tmp_path,callback,message):
    model=torch.nn.Linear(1,1); initial=copy.deepcopy(model.state_dict()); x=torch.ones(8,1)
    with pytest.raises(Exception,match=message):
        distributed.train(model,x,x*2,epochs=3,strategy='fsdp',batch_size=4,output_dir=tmp_path,callback=callback,timeout=30)
    record=json.loads((tmp_path/'shards/latest.json').read_text());assert record['epoch']==1
    assert not (tmp_path/'latest.pt').exists()
    for name,value in model.state_dict().items():torch.testing.assert_close(value,initial[name],rtol=0,atol=0)
    resumed=distributed.train(model,x,x*2,epochs=2,strategy='fsdp',batch_size=4,output_dir=tmp_path/'recover',resume=tmp_path/'shards/latest.json',trusted_resume=True,timeout=30)
    assert len(resumed.history)==2
