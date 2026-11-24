"""Real parallel stepping: heterogeneous spaces, masks, terminal metadata and recurrent state."""
import copy
import random
import numpy as np
import pytest
torch=pytest.importorskip('torch');torch.set_num_threads(1)
gym=pytest.importorskip('gymnasium')
import nil


class MixedParallel:
    def __init__(self,missing_terminal_mask=False):self.missing=missing_terminal_mask
    def observation_space(self,agent):return gym.spaces.Box(-100,100,(1 if agent=='masked' else 2,),dtype=np.float32)
    def action_space(self,agent):return gym.spaces.Discrete(3) if agent=='masked' else gym.spaces.Box(-1,1,(1,),dtype=np.float32)
    def reset(self,seed=None):
        self.count=0
        return {'masked':{'observation':np.array([0.],dtype=np.float32),'action_mask':[1,0,0]},'continuous':np.array([0.,1.],dtype=np.float32)},{'masked':{},'continuous':{}}
    def step(self,actions):
        assert int(actions['masked'])==self.count%3
        assert np.asarray(actions['continuous']).shape==(1,)
        self.count+=1;cut=self.count==2
        # Include reset-looking future observations; terminal metadata is authoritative.
        future={'masked':{'observation':np.array([-99.],dtype=np.float32) if cut else np.array([self.count],dtype=np.float32),
                           'action_mask':[int(i==self.count%3) for i in range(3)]},'continuous':np.array([self.count,1.],dtype=np.float32)}
        infos={'masked':{},'continuous':{}}
        if cut:
            infos['masked']={'terminal_observation':np.array([17.],dtype=np.float32)}
            if not self.missing:infos['masked']['terminal_action_masks']=[0,0,1]
            infos['continuous']={'terminal_observation':np.array([17.,1.],dtype=np.float32)}
        return future,{agent:1. for agent in actions},{agent:False for agent in actions},{agent:cut for agent in actions},infos


class MaskedRecurrent:
    nil_recurrent=True
    def __init__(self):self.calls=[]
    def predict(self,observation,*,state,episode_start,action_masks,deterministic):
        self.calls.append((copy.deepcopy(state),bool(episode_start[0]),np.array(action_masks,copy=True)))
        if episode_start[0]:assert state is None
        return np.int64(np.flatnonzero(action_masks)[0]),(0 if state is None else state)+1


class Continuous:
    def predict(self,obs,**options):return np.array([.25],dtype=np.float32),None


def test_masked_heterogeneous_parallel_collection_preserves_terminal_state_and_rng():
    policy=MaskedRecurrent();random.seed(9);np.random.seed(9);torch.manual_seed(9)
    before=random.getstate(),np.random.get_state(),torch.random.get_rng_state()
    data=nil.rl.collect_multiagent(MixedParallel(),{'masked':policy,'continuous':Continuous()},masked_agents=('masked',),steps=6)
    masked=data['masked'];continuous=data['continuous']
    assert masked['observations'].shape==(6,1) and continuous['observations'].shape==(6,2)
    assert continuous['actions'].shape==(6,1)
    assert masked['actions'].tolist()==[0,1,0,1,0,1]
    assert masked['next_observations'][masked['truncated']].tolist()==[[17.]]*3
    assert masked['next_action_masks'][masked['truncated']].tolist()==[[False,False,True]]*3
    assert [start for _,start,_ in policy.calls]==[True,False]*3
    assert [state for state,_,_ in policy.calls]==[None,1]*3
    assert random.getstate()==before[0]
    np.testing.assert_equal(np.random.get_state(),before[1]);assert torch.equal(torch.random.get_rng_state(),before[2])


def test_truncated_masked_agent_requires_terminal_mask_and_restores_rng():
    before=torch.random.get_rng_state()
    with pytest.raises(ValueError,match='terminal action masks'):
        nil.rl.collect_multiagent(MixedParallel(True),{'masked':MaskedRecurrent(),'continuous':Continuous()},masked_agents=('masked',),steps=2)
    assert torch.equal(torch.random.get_rng_state(),before)


def test_masked_out_action_is_refused_before_environment_step():
    class Bad(MaskedRecurrent):
        def predict(self,*args,**kwargs):return np.int64(2),None
    env=MixedParallel()
    with pytest.raises(ValueError,match='masked-out'):
        nil.rl.collect_multiagent(env,{'masked':Bad(),'continuous':Continuous()},masked_agents=('masked',),steps=2)
    assert env.count==0


def test_truncation_never_uses_an_implicit_reset_observation():
    class MissingFinal(MixedParallel):
        def step(self,actions):
            future,rewards,end,cut,infos=super().step(actions)
            if any(cut.values()):infos['masked'].pop('terminal_observation')
            return future,rewards,end,cut,infos
    with pytest.raises(ValueError,match='terminal_observation'):
        nil.rl.collect_multiagent(MissingFinal(),{'masked':MaskedRecurrent(),'continuous':Continuous()},masked_agents=('masked',),steps=2)
