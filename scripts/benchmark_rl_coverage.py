"""Multi-seed CPU learning evidence on explicitly scoped real Gymnasium tasks.

Offline CQL uses collected random+scripted-expert FrozenLake replay. Offline
centralized-value AWR uses aligned two-CartPole expert trajectories and explicit
team Monte Carlo returns. Neither task certifies broad algorithm superiority.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tempfile
import time


def interval(values):
    import numpy as np
    rng=np.random.default_rng(2026);x=np.asarray(values,dtype=float)
    means=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'bootstrap_95_percent_ci':np.quantile(means,[.025,.975]).tolist()}


def cql(directory,seeds,episodes):
    import gymnasium as gym
    import numpy as np
    import nil
    eye=np.eye(16,dtype='float32');rng=np.random.default_rng(17);env=gym.make('FrozenLake-v1',is_slippery=False);rows=[]
    for episode in range(1256):
        observation,_=env.reset(seed=episode)
        for step in range(50):
            action=int(rng.integers(4)) if episode<1000 else (1,1,2,1,2,2)[step]
            future,reward,end,cut,_=env.step(action)
            rows.append((eye[observation],action,eye[future],reward,end,cut));observation=future
            if end or cut:break
    env.close()
    data={key:np.asarray([row[i] for row in rows]) for i,key in enumerate(('observations','actions','next_observations','rewards','terminated','truncated'))}
    def evaluate(policy,seed):
        environment=gym.make('FrozenLake-v1',is_slippery=False);values=[];random=np.random.default_rng(seed)
        try:
            for episode in range(episodes):
                observation,_=environment.reset(seed=10000+episode);total=0.
                for _ in range(100):
                    action=int(policy.predict(eye[observation])[0]) if policy is not None else int(random.integers(4))
                    observation,reward,end,cut,_=environment.step(action);total+=reward
                    if end or cut:break
                values.append(total)
        finally:environment.close()
        return float(np.mean(values))
    records=[]
    for seed in seeds:
        result=nil.rl.offline_fit(data,algorithm='cql',action_count=4,hidden=(32,),steps=2500,batch_size=128,
            lr=.003,gamma=.95,target_update=20,conservative_weight=.1,seed=seed,output_dir=directory/f'cql-{seed}')
        records.append({'seed':seed,'learned_return':evaluate(result,seed+20000),'random_return':evaluate(None,seed+20000)})
    return {'task':'FrozenLake-v1, fixed4x4 map, is_slippery=False; independent held-out evaluation episodes, not unseen maps',
        'replay':{'random_episodes':1000,'scripted_expert_episodes':256,'transitions':len(rows),'goal_transitions':int(data['rewards'].sum())},
        'updates_per_seed':2500,'records':records}


def centralized(directory,seeds,episodes):
    import gymnasium as gym
    import numpy as np
    import nil
    names=('left','right');observations={name:[] for name in names};actions={name:[] for name in names};returns=[]
    def teacher(x):return int(x[2]+.15*x[3]+.01*x[0]+.01*x[1]>0)
    envs=[gym.make('CartPole-v1') for _ in names]
    try:
        for episode in range(32):
            state=[env.reset(seed=episode*2+i)[0] for i,env in enumerate(envs)];length=0
            for _ in range(500):
                selected=[teacher(x) for x in state]
                for i,name in enumerate(names):observations[name].append(state[i].copy());actions[name].append(selected[i])
                transitions=[env.step(action) for env,action in zip(envs,selected)];length+=1
                state=[item[0] for item in transitions]
                if any(item[2] or item[3] for item in transitions):break
            # Team receives reward1 per joint step until either episode ends.
            returns.extend([(1-.95**(length-index))/(1-.95) for index in range(length)])
    finally:
        for env in envs:env.close()
    datasets={name:{'observations':np.asarray(observations[name],dtype='float32'),'actions':np.asarray(actions[name],dtype='int64')} for name in names}
    def evaluate(policies,seed):
        random=np.random.default_rng(seed);environments=[gym.make('CartPole-v1') for _ in names];scores=[]
        try:
            for episode in range(episodes):
                state=[env.reset(seed=10000+2*episode+i)[0] for i,env in enumerate(environments)]
                for step in range(500):
                    chosen=[int(policies[name].predict(state[i])[0]) if policies else int(random.integers(2)) for i,name in enumerate(names)]
                    transitions=[env.step(action) for env,action in zip(environments,chosen)]
                    state=[item[0] for item in transitions]
                    if any(item[2] or item[3] for item in transitions):break
                scores.append(step+1)
        finally:
            for env in environments:env.close()
        return float(np.mean(scores))
    records=[]
    for seed in seeds:
        model=nil.rl.centralized_fit(datasets,action_counts={name:2 for name in names},returns=returns,steps=750,
            hidden=(32,32),lr=.003,batch_size=128,seed=seed,output_dir=directory/f'centralized-{seed}')
        records.append({'seed':seed,'learned_return':evaluate(model.policies,seed+30000),'random_return':evaluate(None,seed+30000)})
    return {'task':'Fixed two-agent cooperative CartPole-v1 team; local actors, joint-observation value critic; episode stops when either agent ends',
        'algorithm':'Offline centralized-value advantage-weighted regression, explicit discounted team returns; not MAPPO/online exploration',
        'replay':{'expert_joint_episodes':32,'aligned_rows_per_agent':len(returns),'teacher':'sign(angle+.15*angular_velocity+.01*position+.01*velocity)','gamma':.95},
        'updates_per_seed':750,'records':records}


def continuous(directory,seeds,episodes):
    import gymnasium as gym
    import numpy as np
    import nil
    scale=np.array([1.,100.],dtype='float32');observations=[];actions=[]
    env=gym.make('MountainCarContinuous-v0')
    try:
        for episode in range(64):
            state,_=env.reset(seed=episode)
            for _ in range(999):
                action=np.array([1. if state[1]>=0 else -1.],dtype='float32')
                observations.append(state*scale);actions.append(action)
                state,_,end,cut,_=env.step(action)
                if end or cut:break
    finally:env.close()
    data={'observations':np.asarray(observations,dtype='float32'),'actions':np.asarray(actions,dtype='float32')}
    def evaluate(policy,seed):
        environment=gym.make('MountainCarContinuous-v0');random=np.random.default_rng(seed);scores=[]
        try:
            for episode in range(episodes):
                state,_=environment.reset(seed=10000+episode);total=0.
                for _ in range(999):
                    action=policy.predict(state*scale)[0] if policy else random.uniform(-1,1,1).astype('float32')
                    state,reward,end,cut,_=environment.step(action);total+=reward
                    if end or cut:break
                scores.append(total)
        finally:environment.close()
        return float(np.mean(scores))
    records=[]
    for seed in seeds:
        policy=nil.rl.offline_fit(data,hidden=(32,32),steps=750,lr=.01,batch_size=128,seed=seed,
            action_bounds=[[-1.],[1.]],output_dir=directory/f'continuous-{seed}')
        records.append({'seed':seed,'learned_return':evaluate(policy,seed+40000),'random_return':evaluate(None,seed+40000)})
    return {'task':'MountainCarContinuous-v0, CPU offline continuous behavior cloning; explicit observation scale[1,100]',
        'replay':{'expert_episodes':64,'rows':len(actions),'teacher':'action=sign(velocity), positive at zero'},
        'updates_per_seed':750,'records':records}


def masked(directory,seeds,episodes):
    import gymnasium as gym
    import numpy as np
    import nil
    class Context(gym.Env):
        observation_space=gym.spaces.Box(0,1,(3,),dtype=np.float32)
        action_space=gym.spaces.Discrete(3)
        def reset(self,seed=None,options=None):
            super().reset(seed=seed);self.target=int(self.np_random.integers(3))
            self.excluded=(self.target+1+int(self.np_random.integers(2)))%3
            return np.eye(3,dtype='float32')[self.target],{}
        def action_masks(self):return np.arange(3)!=self.excluded
        def step(self,action):
            if not self.action_masks()[int(action)]:raise ValueError('Masked-out action reached environment')
            return np.eye(3,dtype='float32')[self.target],float(action==self.target),True,False,{}
    records=[]
    for seed in seeds:
        policy=nil.rl.create('maskable_ppo',Context(),seed=seed,learning_rate=.01,n_steps=16,batch_size=16,n_epochs=4,policy_kwargs={'net_arch':[16]})
        nil.rl.train(policy,total_timesteps=1024,output_dir=directory/f'masked-{seed}')
        score=nil.rl.evaluate(policy,Context(),episodes=max(100,episodes),max_steps=1,seed=seed+10000)['mean_return']
        env=Context();random=np.random.default_rng(seed+50000);rewards=[]
        for episode in range(max(100,episodes)):
            env.reset(seed=seed+10000+episode);rewards.append(env.step(int(random.choice(np.flatnonzero(env.action_masks()))))[1])
        env.close();policy.get_env().close()
        records.append({'seed':seed,'learned_return':score,'random_return':float(np.mean(rewards))})
    return {'task':'Three-action one-step contextual task, one invalid distractor, random-valid baseline; not sequential masked control',
        'training_steps_per_seed':1024,'evaluation_episodes_per_seed':max(100,episodes),'records':records}


def recurrent(directory,seeds,episodes):
    import gymnasium as gym
    import numpy as np
    import nil
    records=[]
    for seed in seeds:
        policy=nil.rl.create('recurrent_ppo',gym.make('CartPole-v1'),seed=seed,learning_rate=.003,n_steps=32,batch_size=32,n_epochs=3,
            policy_kwargs={'net_arch':[16],'lstm_hidden_size':16})
        nil.rl.train(policy,total_timesteps=8192,output_dir=directory/f'recurrent-{seed}')
        score=nil.rl.evaluate(policy,gym.make('CartPole-v1'),episodes=episodes,max_steps=500,seed=seed+10000)['mean_return']
        env=gym.make('CartPole-v1');random=np.random.default_rng(seed+60000);scores=[]
        for episode in range(episodes):
            env.reset(seed=seed+10000+episode);total=0.
            for _ in range(500):
                _,reward,end,cut,_=env.step(int(random.integers(2)));total+=reward
                if end or cut:break
            scores.append(total)
        env.close();policy.get_env().close()
        records.append({'seed':seed,'learned_return':score,'random_return':float(np.mean(scores))})
    return {'task':'CartPole-v1 with native recurrent PPO; fully observed task, not proof of partially observed memory learning',
        'training_steps_per_seed':8192,'records':records}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',choices=['cql','centralized_awr','continuous','masked','recurrent'],nargs='+',default=['cql','centralized_awr','continuous','masked','recurrent']);parser.add_argument('--episodes',type=int,default=20);parser.add_argument('--seeds',type=int,nargs='+',default=[3,7,11])
    args=parser.parse_args()
    if not 3<=len(set(args.seeds))==len(args.seeds)<=10 or any(not 0<=seed<2**32 for seed in args.seeds) or not 10<=args.episodes<=100:parser.error('Use 3–10 unique seeds and 10–100 evaluation episodes')
    import torch
    torch.set_num_threads(1);started=time.monotonic()
    report={'schema_version':1,'status':'failed','platform':platform.platform(),'python':platform.python_version(),
        'versions':{name:version(name) for name in ('torch','numpy','gymnasium','stable-baselines3','sb3-contrib')},'evaluation_episodes_per_seed':args.episodes,'cases':{}}
    with tempfile.TemporaryDirectory(prefix='nil-rl-coverage-') as temporary:
        for name in args.cases:
            run={'cql':cql,'centralized_awr':centralized,'continuous':continuous,'masked':masked,'recurrent':recurrent}[name]
            result=run(Path(temporary),args.seeds,args.episodes)
            improvements=[row['learned_return']-row['random_return'] for row in result['records']]
            result['paired_improvement']=interval(improvements);result['status']='passed' if result['paired_improvement']['bootstrap_95_percent_ci'][0]>0 else 'failed'
            report['cases'][name]=result
    report['status']='passed' if all(value['status']=='passed' for value in report['cases'].values()) else 'failed'
    report['seconds']=time.monotonic()-started;report['scope']='Task-specific held-out CPU learning; seeds are sampling units; no generalization to new maps/environments or broad RL quality claim'
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'seconds':report['seconds'],'cases':{name:value['paired_improvement'] for name,value in report['cases'].items()}}))
    return 0 if report['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
