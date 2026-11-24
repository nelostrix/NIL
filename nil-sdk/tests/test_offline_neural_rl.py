import numpy as np
import pytest
torch=pytest.importorskip('torch'); torch.set_num_threads(1)
import nil


def test_neural_behavior_cloning_heldout_and_exact_resume(tmp_path):
    rng=np.random.default_rng(7); x=rng.normal(size=(128,2)).astype('float32'); actions=(x[:,0]+x[:,1]>0).astype('int64')
    data={'observations':x,'actions':actions}
    options=dict(action_count=2,hidden=(16,),lr=.02,batch_size=32,seed=7)
    full=nil.rl.offline_fit(data,steps=120,output_dir=tmp_path/'full',**options)
    info=nil.contracts.result_info(full)
    assert info.family=='rl.offline' and info.history_length==len(full.history) and info.model_count==1
    first=nil.rl.offline_fit(data,steps=40,output_dir=tmp_path/'first',**options)
    resumed=nil.rl.offline_fit(data,steps=120,resume=first.checkpoint,output_dir=tmp_path/'resume',**options)
    assert full.history==resumed.history
    for left,right in zip(full.model.policy.parameters(),resumed.model.policy.parameters()): torch.testing.assert_close(left,right,rtol=0,atol=0)
    test=rng.normal(size=(100,2)).astype('float32'); predicted,_=nil.predict(full,test)
    assert (predicted==(test.sum(axis=1)>0)).mean()>.9
    saved=nil.rl.save(full,tmp_path/'policy'); loaded=nil.rl.load(saved,trusted=True)
    np.testing.assert_array_equal(loaded.predict(test)[0],predicted)
    with pytest.raises(ValueError,match='same dataset'): nil.rl.offline_fit({**data,'actions':1-actions},steps=121,resume=first.checkpoint,output_dir=tmp_path/'bad',**options)


def test_offline_dqn_terminal_truncation_masks_and_learning(tmp_path):
    observations=np.zeros((128,1),dtype='float32'); actions=np.tile([0,1],64)
    data={'observations':observations,'actions':actions,'next_observations':observations.copy(),'rewards':actions.astype('float32'),'terminated':np.ones(128,dtype=bool),'truncated':np.zeros(128,dtype=bool),'next_action_masks':np.zeros((128,2),dtype=bool)}
    result=nil.rl.offline_fit(data,algorithm='offline_dqn',action_count=2,steps=160,hidden=(8,),lr=.02,output_dir=tmp_path/'dqn')
    assert result.model.predict([0])[0]==1
    assert result.model.predict([0],action_masks=[True,False])[0]==0
    with pytest.raises(ValueError,match='Nonterminal'): nil.rl.offline_fit({**data,'terminated':np.zeros(128,dtype=bool),'truncated':np.ones(128,dtype=bool)},algorithm='offline_dqn',action_count=2,steps=1,output_dir=tmp_path/'bad')
    with pytest.raises(ValueError,match='masked-out'): nil.rl.offline_fit({'observations':observations,'actions':actions,'action_masks':np.tile([True,False],(128,1))},action_count=2,steps=1,output_dir=tmp_path/'invalid')


def test_continuous_neural_imitation_and_dataset_limits(tmp_path):
    x=np.linspace(-1,1,64,dtype='float32')[:,None]; y=x*.5
    result=nil.rl.offline_fit({'observations':x,'actions':y},steps=150,hidden=(8,),lr=.02,action_bounds=[[-1],[1]],output_dir=tmp_path/'continuous')
    assert np.mean((result.model.predict(x)[0]-y)**2)<.01
    with pytest.raises(ValueError,match='max_dataset_bytes'): nil.rl.offline_fit({'observations':x,'actions':y},steps=1,max_dataset_bytes=10,output_dir=tmp_path/'large')


def test_dataset_budget_counts_normalized_storage_before_conversion(tmp_path):
    with pytest.raises(ValueError,match='Normalized offline dataset exceeds max_dataset_bytes'):
        nil.rl.offline_fit({'observations':torch.ones(2,1,dtype=torch.float16),
            'actions':torch.ones(2,1,dtype=torch.float16)},max_dataset_bytes=12,
            steps=1,output_dir=tmp_path)
