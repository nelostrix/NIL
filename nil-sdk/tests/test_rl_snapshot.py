"""Real stochastic simulator/replay/optimizer continuation at safe CPU boundaries."""
import copy
import threading
import pytest
pytest.importorskip('stable_baselines3')
import gymnasium as gym
import numpy as np
import torch
import nil

class Stateful(gym.Env):
    observation_space=gym.spaces.Box(-100,100,(1,),dtype=np.float32)
    action_space=gym.spaces.Discrete(2)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed); self.position=float(self.np_random.uniform(-1,1)); self.steps=0
        return np.array([self.position],np.float32),{}
    def step(self,action):
        self.position+=float(action)*.1+float(self.np_random.uniform(-.01,.01)); self.steps+=1
        return np.array([self.position],np.float32),float(action==1),False,self.steps>=7,{}
    def state_dict(self): return {'position':self.position,'steps':self.steps}
    def validate_state_dict(self,state):
        if set(state)!={'position','steps'} or not np.isfinite(state['position']) or type(state['steps']) is not int or not 0<=state['steps']<=7: raise ValueError('Invalid simulator state')
    def load_state_dict(self,state): self.validate_state_dict(state); self.position=state['position']; self.steps=state['steps']

def agent():
    return nil.rl.create('dqn',Stateful(),seed=7,buffer_size=128,batch_size=8,learning_starts=4,
        train_freq=4,gradient_steps=1,policy_kwargs={'net_arch':[8]},learning_rate=.001)

def test_exact_mid_episode_resume_with_replay_and_rng(tmp_path):
    model=agent(); nil.rl.train(model,total_timesteps=20,output_dir=tmp_path/'first')
    bundle=nil.rl.save(model,tmp_path/'bundle',environment_state=True)
    nil.rl.train(model,total_timesteps=24,output_dir=tmp_path/'resident')
    expected=copy.deepcopy(model.policy.state_dict()); replay=copy.deepcopy(model.replay_buffer)
    restored=nil.rl.load(bundle,env=Stateful(),trusted=True,restore_rng=True,restore_environment=True)
    nil.rl.train(restored,total_timesteps=24,output_dir=tmp_path/'restored')
    for name,value in expected.items(): torch.testing.assert_close(restored.policy.state_dict()[name],value,rtol=0,atol=0)
    assert restored.replay_buffer.pos==replay.pos
    for name in ('observations','next_observations','actions','rewards','dones'):
        # Only committed replay rows have defined values.
        np.testing.assert_array_equal(getattr(restored.replay_buffer,name)[:replay.pos],getattr(replay,name)[:replay.pos])
    assert restored.get_env().envs[0].env.state_dict()==model.get_env().envs[0].env.state_dict()
    with pytest.raises(ValueError,match='requires env'): nil.rl.load(bundle,trusted=True,restore_environment=True)
    model.get_env().close(); restored.get_env().close()

def test_cancel_commits_complete_collection_and_rejects_unknown_simulator(tmp_path):
    model=agent(); cancelled=threading.Event()
    def callback(locals,globals):
        if locals['self'].num_timesteps>=3: cancelled.set()
        return True
    result=nil.rl.train(model,total_timesteps=20,cancel_event=cancelled,callback=callback,output_dir=tmp_path/'runs')
    assert result.status=='cancelled' and result.actual_steps==4
    assert model.replay_buffer.size()==4
    nil.rl.save(model,tmp_path/'cancelled',environment_state=True)
    # A callback returning False aborts before native transition commit.
    nil.rl.train(model,total_timesteps=4,callback=lambda *_:False,output_dir=tmp_path/'unsafe')
    with pytest.raises(ValueError,match='boundary'): nil.rl.save(model,tmp_path/'unsafe-save',environment_state=True)
    model.get_env().close()
    other=nil.rl.create('dqn',gym.make('CartPole-v1'),buffer_size=32,learning_starts=32,train_freq=1)
    nil.rl.train(other,total_timesteps=1,output_dir=tmp_path/'unsupported')
    with pytest.raises(ValueError,match='implement'): nil.rl.save(other,tmp_path/'unsupported-save',environment_state=True)
    other.get_env().close()
