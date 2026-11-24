"""Discrete conservative offline Q: OOD actions, exact resume, masks and real learning."""
import numpy as np
import pytest
torch=pytest.importorskip('torch');torch.set_num_threads(1)
import nil


def dataset():
    x=np.linspace(-1,1,128,dtype='float32')[:,None]
    return {'observations':x,'actions':np.zeros(128,dtype='int64'),'next_observations':x.copy(),
            'rewards':-np.ones(128,dtype='float32'),'terminated':np.ones(128,dtype=bool),'truncated':np.zeros(128,dtype=bool)}


def test_cql_suppresses_unobserved_actions_and_has_exact_resume(tmp_path):
    data=dataset();options=dict(algorithm='cql',action_count=3,hidden=(8,),lr=.02,batch_size=32,seed=7,conservative_weight=1.)
    full=nil.rl.offline_fit(data,steps=150,output_dir=tmp_path/'full',**options)
    first=nil.rl.offline_fit(data,steps=50,output_dir=tmp_path/'first',**options)
    resumed=nil.rl.offline_fit(data,steps=150,resume=first.checkpoint,output_dir=tmp_path/'resumed',**options)
    assert full.history==resumed.history
    for a,b in zip(full.model.policy.parameters(),resumed.model.policy.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    heldout=np.random.default_rng(17).uniform(-.9,.9,(100,1)).astype('float32')
    assert (full.predict(heldout)[0]==0).mean()==1.
    with torch.no_grad():
        q=full.model.policy(torch.as_tensor(heldout))
        assert (q[:,0,None]>q[:,1:]).all()
    restored=nil.rl.load(nil.rl.save(full,tmp_path/'bundle'),trusted=True)
    np.testing.assert_equal(restored.predict(heldout)[0],full.predict(heldout)[0])
    with pytest.raises(ValueError,match='same dataset/config'):
        nil.rl.offline_fit(data,steps=151,resume=first.checkpoint,output_dir=tmp_path/'changed',**{**options,'conservative_weight':.5})


def test_cql_action_masks_and_truncated_bootstrap_guards(tmp_path):
    data=dataset();data['action_masks']=np.tile([True,False,False],(128,1));data['next_action_masks']=np.zeros((128,3),dtype=bool)
    result=nil.rl.offline_fit(data,algorithm='cql',action_count=3,steps=10,output_dir=tmp_path/'masks')
    assert np.isfinite(result.history).all()
    with pytest.raises(ValueError,match='Nonterminal'):
        nil.rl.offline_fit({**data,'terminated':np.zeros(128,dtype=bool),'truncated':np.ones(128,dtype=bool)},algorithm='cql',action_count=3,steps=1,output_dir=tmp_path/'bad')
    with pytest.raises(ValueError,match='conservative_weight'):
        nil.rl.offline_fit(dataset(),algorithm='cql',action_count=3,conservative_weight=float('nan'),output_dir=tmp_path/'nan')
