"""Small reproducible learning-quality gate; no general algorithm quality claim.

Run from the repository with optional NIL rl backends installed:
  PYTHONPATH=nil-sdk python scripts/benchmark_rl.py --output report.json
"""
import argparse
import json
import platform
import random
import tempfile
from importlib import metadata
from pathlib import Path
import time


def interval(values,seed=2026):
    import numpy as np
    values=np.asarray(values,dtype=float)
    if len(values)<2: raise ValueError('At least two independent seeds are required')
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
    return {'mean':float(values.mean()),'bootstrap_95_percent_ci':[float(x) for x in np.quantile(means,[.025,.975])]}


def run(seeds=(0,1,2,3,4),steps=512,episodes=100):
    import gymnasium as gym
    import numpy as np
    import torch
    import nil
    torch.set_num_threads(1)
    class Bandit(gym.Env):
        observation_space=gym.spaces.Box(-1,1,(1,),dtype=np.float32)
        action_space=gym.spaces.Discrete(2)
        def reset(self,*,seed=None,options=None):
            super().reset(seed=seed); return np.zeros(1,np.float32),{}
        def step(self,action): return np.zeros(1,np.float32),float(action==1),True,False,{}
    records=[]; start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='nil-quality-') as directory:
        for seed in seeds:
            model=nil.rl.create('ppo',Bandit(),seed=seed,learning_rate=.01,n_steps=16,batch_size=16,n_epochs=2,policy_kwargs={'net_arch':[16,16]})
            nil.rl.train(model,total_timesteps=steps,output_dir=directory)
            score=nil.rl.evaluate(model,Bandit(),episodes=episodes,max_steps=1,seed=seed)['mean_return']
            rng=random.Random(seed+10000)
            baseline_env=Bandit(); baseline_env.reset(seed=seed)
            rewards=[baseline_env.step(rng.randrange(2))[1] for _ in range(episodes)]
            baseline_env.close(); model.get_env().close()
            records.append({'seed':seed,'policy_mean_return':score,'random_mean_return':sum(rewards)/episodes})
    trained=[row['policy_mean_return'] for row in records]; baseline=[row['random_mean_return'] for row in records]
    return {'task':'one-step two-action deterministic bandit, reward=1(action==1)',
        'scope':'CPU PPO integration learning sanity on this task only; seeds, not episodes, are the CI sampling units',
        'training_steps_per_seed':steps,'evaluation_episodes_per_seed':episodes,'records':records,
        'trained':interval(trained),'random_baseline':interval(baseline),
        'paired_improvement':interval([x-y for x,y in zip(trained,baseline)]),
        'seconds':time.perf_counter()-start,'platform':platform.platform(),'processor':platform.processor(),
        'python':platform.python_version(),'versions':{name:metadata.version(name) for name in ('torch','stable-baselines3','gymnasium','numpy')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); report=run(); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({key:report[key] for key in ('trained','random_baseline','paired_improvement','seconds')}))
    return 0 if report['paired_improvement']['bootstrap_95_percent_ci'][0]>.1 else 1

if __name__=='__main__': raise SystemExit(main())
