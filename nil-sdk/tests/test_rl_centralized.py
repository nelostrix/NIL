"""Fixed-cohort offline centralized-value AWR: held-out policies, exact recovery and budgets."""
import numpy as np
import pytest
torch=pytest.importorskip('torch');torch.set_num_threads(1)
import nil


def data():
    rng=np.random.default_rng(7);x=rng.normal(size=(128,2)).astype('float32')
    sets={'a':{'observations':x[:,:1],'actions':(x[:,0]>0).astype('int64')},
          'b':{'observations':x,'actions':(x[:,0]+x[:,1]>0).astype('int64')}}
    return sets,np.full(128,2.,dtype='float32')


def test_centralized_awr_heldout_and_exact_resume(tmp_path):
    sets,returns=data();options=dict(action_counts={'a':2,'b':2},returns=returns,hidden=(16,),lr=.02,batch_size=32,seed=7)
    full=nil.rl.centralized_fit(sets,steps=150,output_dir=tmp_path/'full',**options)
    info=nil.contracts.result_info(full)
    assert info.family=='rl.centralized' and info.model_count==2 and info.history_length==len(full.history)
    first=nil.rl.centralized_fit(sets,steps=50,output_dir=tmp_path/'first',**options)
    resumed=nil.rl.centralized_fit(sets,steps=150,resume=first.checkpoint,output_dir=tmp_path/'resume',**options)
    assert full.history==resumed.history
    test=np.random.default_rng(19).normal(size=(100,2)).astype('float32')
    for name,observations,target in (('a',test[:,:1],test[:,0]>0),('b',test,test.sum(1)>0)):
        np.testing.assert_equal(full.policies[name].predict(observations)[0],resumed.policies[name].predict(observations)[0])
        assert (full.policies[name].predict(observations)[0]==target).mean()>.9
    for a,b in zip(full.critic.parameters(),resumed.critic.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    loaded=nil.rl.centralized_load(full.checkpoint,trusted=True)
    np.testing.assert_equal(loaded.policies['b'].predict(test)[0],full.policies['b'].predict(test)[0])
    with pytest.raises(ValueError,match='trusted'):nil.rl.centralized_load(full.checkpoint)
    with pytest.raises(ValueError,match='identical'):
        nil.rl.centralized_fit(sets,steps=151,resume=first.checkpoint,output_dir=tmp_path/'bad',**{**options,'returns':returns+1})


def test_centralized_awr_masks_and_input_contracts(tmp_path):
    sets,returns=data();sets['a']['action_masks']=np.ones((128,2),dtype=bool)
    result=nil.rl.centralized_fit(sets,action_counts={'a':2,'b':2},returns=returns,steps=2,output_dir=tmp_path/'valid')
    assert result.policies['a'].predict([0.],action_masks=[False,True])[0]==1
    with pytest.raises(ValueError,match='Joint dataset'):
        nil.rl.centralized_fit(sets,action_counts={'a':2,'b':2},returns=returns,max_dataset_bytes=3000,steps=1,output_dir=tmp_path/'budget')
    with pytest.raises(ValueError,match='aligned'):
        nil.rl.centralized_fit({**sets,'b':{'observations':sets['b']['observations'][:2],'actions':sets['b']['actions'][:2]}},action_counts={'a':2,'b':2},returns=returns,steps=1,output_dir=tmp_path/'rows')
