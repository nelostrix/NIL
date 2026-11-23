import numpy as np
import pytest
torch=pytest.importorskip('torch'); torch.set_num_threads(1)
gym=pytest.importorskip('gymnasium')
pytest.importorskip('stable_baselines3')
import nil


class MaskEnv(gym.Env):
    observation_space=gym.spaces.Box(0,1,(1,),dtype=np.float32); action_space=gym.spaces.Discrete(2)
    def reset(self,seed=None,options=None): super().reset(seed=seed); self.count=0; return np.zeros(1,dtype=np.float32),{}
    def action_masks(self): return np.array([self.count%2==0,self.count%2==1])
    def step(self,action):
        assert self.action_masks()[int(action)]
        self.count+=1
        return np.array([self.count%2],dtype=np.float32),1.,self.count==3,False,{}


class EndEnv(gym.Env):
    observation_space=gym.spaces.Box(-100,100,(1,),dtype=np.float32); action_space=gym.spaces.Discrete(2)
    def reset(self,seed=None,options=None): super().reset(seed=seed); self.count=0; return np.array([-99],dtype=np.float32),{}
    def step(self,action): self.count+=1; return np.array([7],dtype=np.float32),1.,False,True,{}


@pytest.mark.parametrize('algorithm',['recurrent_ppo','maskable_ppo'])
def test_optional_native_recurrent_and_masked_train_evaluate_roundtrip(tmp_path,algorithm):
    pytest.importorskip('sb3_contrib')
    env=MaskEnv() if algorithm=='maskable_ppo' else gym.make('CartPole-v1')
    model=nil.rl.create(algorithm,env,n_steps=8,batch_size=8,n_epochs=1,seed=7)
    trained=nil.rl.train(model,total_timesteps=32,output_dir=tmp_path/'runs')
    report=nil.rl.evaluate(trained,MaskEnv() if algorithm=='maskable_ppo' else gym.make('CartPole-v1'),episodes=2,max_steps=30)
    assert report['total_steps']>0
    if algorithm=='maskable_ppo': assert report['mean_return']==3
    loaded=nil.rl.load(nil.rl.save(model,tmp_path/'policy'),trusted=True)
    options={'action_masks':[True,False]} if algorithm=='maskable_ppo' else {'episode_start':np.ones(1,dtype=bool)}
    observation=np.zeros(1 if algorithm=='maskable_ppo' else 4,dtype=np.float32)
    np.testing.assert_array_equal(loaded.predict(observation,deterministic=True,**options)[0],model.predict(observation,deterministic=True,**options)[0])
    model.get_env().close()


def test_vector_autoreset_uses_terminal_observation_and_truncation():
    from stable_baselines3.common.vec_env import DummyVecEnv
    class Policy:
        def predict(self,x,**options): return np.zeros(len(x),dtype=int),None
    env=DummyVecEnv([EndEnv,EndEnv])
    try:
        data=nil.rl.collect_vector(Policy(),env,steps=2)
        assert data['observations'].shape==(4,1)
        assert (data['next_observations']==7).all() and (data['observations']==-99).all()
        assert data['truncated'].all() and not data['terminated'].any()
    finally: env.close()


class ParallelEnv:
    def reset(self,seed=None): self.count=0; return {'a':np.array([0.,1.]),'b':np.array([0.,1.])},{'a':{},'b':{}}
    def step(self,actions):
        self.count+=1; end=self.count==2
        obs={agent:np.array([self.count%2,1.]) for agent in actions}
        return obs,{agent:float(action==(0 if agent=='a' else 1)) for agent,action in actions.items()},{agent:end for agent in actions},{agent:False for agent in actions},{agent:{} for agent in actions}


def test_multiagent_episode_contract_and_actual_neural_independent_imitation(tmp_path):
    class Teacher:
        def __init__(self,action): self.action=action
        def predict(self,obs,**options): return self.action,None
    data=nil.rl.collect_multiagent(ParallelEnv(),{'a':Teacher(0),'b':Teacher(1)},steps=10)
    assert data['a']['terminated'].sum()==5
    models=nil.rl.multiagent_fit(data,action_count=2,steps=50,hidden=(8,),lr=.03,output_dir=tmp_path)
    assert models['a'].predict([0.,1.])[0]==0 and models['b'].predict([0.,1.])[0]==1
