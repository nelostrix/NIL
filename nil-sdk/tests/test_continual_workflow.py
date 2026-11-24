"""Held-out continual classification, replay and stage transaction recovery."""
import copy
import json
import pytest
import torch
import nil


def tasks():
    generator=torch.Generator().manual_seed(7)
    result=[]
    for stage in range(2):
        train=torch.randn(64,2,generator=generator);test=torch.randn(64,2,generator=generator)
        # Explicit task context disambiguates the different labeling rules.
        # Held-out labels never enter fitting or replay.
        train_y=(train[:,stage]>0).long();test_y=(test[:,stage]>0).long()
        train=torch.cat([train,torch.full((64,1),float(2*stage-1))],1)
        test=torch.cat([test,torch.full((64,1),float(2*stage-1))],1)
        result.append(dict(id=f'task{stage}',train=(train,train_y),test=(test,test_y)))
    return result


def model():
    torch.manual_seed(11)
    return torch.nn.Sequential(torch.nn.Linear(3,12),torch.nn.Tanh(),torch.nn.Linear(12,2))


def test_replay_training_exact_stage_resume_and_heldout_matrix(tmp_path):
    data=tasks();options=dict(classes=2,epochs=10,batch_size=16,learning_rate=.03,replay_capacity=32,seed=9)
    full=nil.learning.continual_fit(model(),data,output_dir=tmp_path/'full',**options)
    partial=nil.learning.continual_fit(model(),data,output_dir=tmp_path/'resume',stop_after=1,**options)
    assert partial['status']=='paused' and partial['accuracy'][0][1] is None
    resumed=nil.learning.continual_fit(model(),data,output_dir=tmp_path/'resume',resume=True,**options)
    assert resumed['status']=='completed' and resumed['accuracy']==full['accuracy']
    assert resumed['replay']==full['replay'] and resumed['replay']['seen']==128
    assert resumed['accuracy'][0][0]>.85 and resumed['accuracy'][1][1]>.8
    for key,value in full['model'].state_dict().items():torch.testing.assert_close(value,resumed['model'].state_dict()[key],rtol=0,atol=0)
    snapshot=nil.learning.ReplayMemory.from_state_dict(resumed['replay']);owned=snapshot.state_dict();owned['records'].clear()
    assert snapshot.records()


def test_changed_data_config_or_architecture_rejected_before_mutation(tmp_path):
    data=tasks();options=dict(classes=2,epochs=1,replay_capacity=16)
    nil.learning.continual_fit(model(),data,output_dir=tmp_path/'run',stop_after=1,**options)
    target=model();before=copy.deepcopy(target.state_dict());rng=torch.get_rng_state().clone()
    changed=tasks();changed[0]['test'][0][0,0]+=1
    for candidate,config in [(changed,options),(data,dict(options,replay_capacity=17))]:
        with pytest.raises(ValueError,match='mismatch'):nil.learning.continual_fit(target,candidate,output_dir=tmp_path/'run',resume=True,**config)
        assert torch.equal(rng,torch.get_rng_state())
        for key,value in before.items():torch.testing.assert_close(value,target.state_dict()[key],rtol=0,atol=0)
    wrong=torch.nn.Linear(3,2);before=copy.deepcopy(wrong.state_dict());rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='structure'):nil.learning.continual_fit(wrong,data,output_dir=tmp_path/'run',resume=True,**options)
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in before.items():torch.testing.assert_close(value,wrong.state_dict()[key],rtol=0,atol=0)


def test_stage_publication_failure_resumes_last_commit(tmp_path,monkeypatch):
    import nil._continual as implementation
    data=tasks();options=dict(classes=2,epochs=2,replay_capacity=16,seed=8)
    expected=nil.learning.continual_fit(model(),data,output_dir=tmp_path/'reference',**options)
    original=implementation.atomic_json
    def crash(path,value):
        if path.name=='state.json' and value['completed']==2:raise OSError('injected commit failure')
        return original(path,value)
    monkeypatch.setattr(implementation,'atomic_json',crash)
    with pytest.raises(OSError,match='injected'):nil.learning.continual_fit(model(),data,output_dir=tmp_path/'run',**options)
    assert json.loads((tmp_path/'run/state.json').read_text())['completed']==1
    monkeypatch.setattr(implementation,'atomic_json',original)
    actual=nil.learning.continual_fit(model(),data,output_dir=tmp_path/'run',resume=True,**options)
    assert actual['accuracy']==expected['accuracy'] and actual['replay']==expected['replay']
    for key,value in expected['model'].state_dict().items():torch.testing.assert_close(value,actual['model'].state_dict()[key],rtol=0,atol=0)
