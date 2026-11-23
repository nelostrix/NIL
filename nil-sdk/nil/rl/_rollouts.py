"""Structured CPU rollout contracts; terminal observations never become reset observations."""
from contextlib import contextmanager
import random
from pathlib import Path
from nil.runtime import require
from nil._classical_common import positive_int


def action_mask(env):
    np=require('numpy','rl')
    getter=env.get_wrapper_attr('action_masks') if hasattr(env,'get_wrapper_attr') else getattr(env,'action_masks',None)
    if not callable(getter): raise ValueError('Masked evaluation requires env.action_masks()')
    values=np.asarray(getter())
    if values.shape!=(env.action_space.n,) or not np.isin(values,[0,1]).all() or not values.any(): raise ValueError('Invalid or empty action mask')
    return values.astype(bool)


@contextmanager
def _evaluation_context(policies,seed):
    np=require('numpy','rl'); torch=require('torch','torch')
    previous=random.getstate(),np.random.get_state(); modes=[]; rngs=[]
    for model in policies:
        if getattr(model,'device',torch.device('cpu')).type!='cpu': raise ValueError('Structured rollout acceptance currently supports CPU policies')
        if hasattr(model,'policy'): modes.append((model.policy,model.policy.training))
        if hasattr(model,'action_generator'): rngs.append((model.action_generator,model.action_generator.get_state()))
    try:
        with torch.random.fork_rng():
            torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
            yield
    finally:
        random.setstate(previous[0]); np.random.set_state(previous[1])
        for policy,mode in modes:
            if hasattr(policy,'set_training_mode'):policy.set_training_mode(mode)
            else:policy.train(mode)
        for generator,state in rngs: generator.set_state(state)


def _predict(policy,observation,state,episode_start,deterministic,mask=None):
    np=require('numpy','rl'); options={'deterministic':deterministic}
    if type(policy).__name__=='RecurrentPPO' or getattr(policy,'nil_recurrent',False): options.update(state=state,episode_start=np.asarray(episode_start))
    if mask is not None: options['action_masks']=mask
    return policy.predict(observation,**options)


def collect_vector(policy,env,*,steps=100,seed=0,deterministic=False,max_transitions=10000):
    np=require('numpy','rl'); vector=require('stable_baselines3.common.vec_env','rl')
    positive_int(steps,'steps'); positive_int(max_transitions,'max_transitions')
    if not isinstance(env,vector.DummyVecEnv) or not getattr(env.observation_space,'shape',None) or len(env.observation_space.shape)!=1: raise ValueError('Use a separate DummyVecEnv with flat Box observations')
    if steps*env.num_envs>max_transitions: raise ValueError('Vector collection exceeds max_transitions')
    training=getattr(policy,'get_env',lambda:None)()
    while hasattr(training,'venv'): training=training.venv
    if hasattr(training,'envs') and any(a.unwrapped is b.unwrapped for a in training.envs for b in env.envs): raise ValueError('Use a separate collection environment')
    rows=[]; state=None; starts=np.ones(env.num_envs,dtype=bool)
    with _evaluation_context([policy],seed):
        env.seed(seed); observations=env.reset()
        for _ in range(steps):
            masks=np.asarray([action_mask(item) for item in env.envs]) if type(policy).__name__=='MaskablePPO' else None
            actions,state=_predict(policy,observations,state,starts,deterministic,masks)
            future,rewards,dones,infos=env.step(actions)
            for i in range(env.num_envs):
                truncated=bool(infos[i].get('TimeLimit.truncated',False)); terminated=bool(dones[i] and not truncated)
                next_observation=infos[i].get('terminal_observation') if dones[i] else future[i]
                if next_observation is None: raise ValueError('Auto-reset transitions require terminal_observation')
                row={'observations':np.array(observations[i],copy=True),'actions':np.array(actions[i],copy=True),'next_observations':np.array(next_observation,copy=True),'rewards':float(rewards[i]),'terminated':terminated,'truncated':truncated}
                if masks is not None:
                    row['action_masks']=masks[i]
                    if dones[i]:
                        next_mask=infos[i].get('terminal_action_masks')
                        if next_mask is None and not terminated: raise ValueError('Truncated masked transitions require terminal_action_masks')
                        row['next_action_masks']=np.zeros_like(masks[i]) if next_mask is None else next_mask
                    else: row['next_action_masks']=action_mask(env.envs[i])
                rows.append(row)
            observations=future; starts=dones
    result={key:np.asarray([row[key] for row in rows]) for key in rows[0]}
    if not all(np.isfinite(value).all() for value in result.values()): raise ValueError('Nonfinite collected transitions')
    return result


def collect_multiagent(env,policies,*,steps=100,seed=0,deterministic=True,max_transitions=10000,masked_agents=()):
    np=require('numpy','rl');positive_int(steps,'steps');positive_int(max_transitions,'max_transitions')
    if not isinstance(policies,dict) or not policies or len(policies)>128:raise ValueError('Provide 1–128 explicit agent policies')
    if not isinstance(masked_agents,(list,tuple,set)) or not set(masked_agents)<=set(policies):raise ValueError('Unknown masked agent')
    masked=set(masked_agents)|{agent for agent,policy in policies.items() if type(policy).__name__=='MaskablePPO'}
    if any(type(policies[agent]).__name__=='RecurrentPPO' for agent in masked):raise ValueError('RecurrentPPO does not implement masked prediction; provide a declared compatible policy')
    def observation(value):
        raw=value.get('observation') if isinstance(value,dict) else value
        array=np.asarray(raw)
        if not array.size or not np.issubdtype(array.dtype,np.number) or not np.isfinite(array).all():raise ValueError('Expected finite per-agent numeric observations')
        return np.array(array,copy=True)
    def mask_for(agent,raw,info,*,terminal=False,terminated=False,expected_count=None):
        space=getattr(env,'action_space',lambda _:None)(agent)
        if space is None:space=getattr(policies[agent],'action_space',None)
        count=getattr(space,'n',None) if space is not None else expected_count
        supplied=info.get('terminal_action_masks') if terminal else None
        if supplied is None and isinstance(raw,dict):supplied=raw.get('action_mask')
        if supplied is None and not terminal:supplied=info.get('action_mask')
        if supplied is None and not terminal and callable(getattr(env,'action_masks',None)):supplied=env.action_masks(agent)
        if supplied is None and terminated and count is not None:supplied=np.zeros(count,dtype=bool)
        values=np.asarray(supplied)
        if values.ndim!=1 or not 1<=len(values)<=4096 or (count is not None and values.shape!=(count,)) or not np.isin(values,[0,1]).all() or (not terminated and not values.any()):
            raise ValueError('Masked transitions require valid current/terminal action masks, including truncations')
        return values.astype(bool,copy=True)
    rows={agent:[] for agent in policies};states={};episode=0;transitions=0
    with _evaluation_context(list(policies.values()),seed):
        observations,current_infos=env.reset(seed=seed)
        for _ in range(steps):
            if not observations:
                episode+=1;observations,current_infos=env.reset(seed=seed+episode);states={}
            if not isinstance(observations,dict) or not observations or not set(observations)<=set(policies):raise ValueError('Unknown agents or invalid parallel observations')
            if not isinstance(current_infos,dict):raise ValueError('Parallel reset/step info must be agent-keyed')
            if transitions+len(observations)>max_transitions:raise ValueError('Multi-agent transition budget exceeded')
            arrays={agent:observation(value) for agent,value in observations.items()}
            masks={agent:mask_for(agent,observations[agent],current_infos.get(agent,{})) for agent in observations if agent in masked}
            actions={}
            for agent,value in arrays.items():
                space=getattr(env,'observation_space',lambda _:None)(agent)
                if space is not None and not space.contains(value):raise ValueError('Observation does not match declared agent space')
                actions[agent],states[agent]=_predict(policies[agent],value,states.get(agent),[agent not in states],deterministic,masks.get(agent))
                action_space=getattr(env,'action_space',lambda _:None)(agent)
                if action_space is not None and not action_space.contains(actions[agent]):raise ValueError('Policy action does not match declared agent space')
                if agent in masks:
                    action=np.asarray(actions[agent])
                    if action.ndim or not np.issubdtype(action.dtype,np.integer) or not 0<=int(action)<len(masks[agent]) or not masks[agent][int(action)]:raise ValueError('Policy selected a masked-out action')
            future,rewards,terminated,truncated,infos=env.step(actions)
            if not all(isinstance(mapping,dict) for mapping in (future,rewards,terminated,truncated,infos)) or not all(set(observations)<=set(mapping) for mapping in (rewards,terminated,truncated,infos)) or not set(future)<=set(policies):raise ValueError('Parallel dictionaries must describe every acting agent and no unknown future agents')
            for agent,value in arrays.items():
                end,cut=terminated[agent],truncated[agent]
                if type(end) not in (bool,np.bool_) or type(cut) not in (bool,np.bool_):raise ValueError('Agent termination flags must be boolean')
                info=infos[agent]
                if not isinstance(info,dict):raise ValueError('Agent info must be a dictionary')
                if cut and info.get('terminal_observation') is None:raise ValueError('Truncated parallel agents require terminal_observation')
                raw=info.get('terminal_observation',future.get(agent)) if end or cut else future.get(agent)
                if raw is None:
                    if not end:raise ValueError('Truncated/nonterminal agents need a next observation')
                    next_observation=np.zeros_like(value)
                else:next_observation=observation(raw)
                if next_observation.shape!=value.shape:raise ValueError('Per-agent observation shape changed')
                row={'observations':value,'actions':np.array(actions[agent],copy=True),'next_observations':next_observation,
                     'rewards':float(rewards[agent]),'terminated':bool(end),'truncated':bool(cut)}
                if agent in masks:
                    row['action_masks']=masks[agent]
                    row['next_action_masks']=mask_for(agent,raw,info,terminal=bool(end or cut),terminated=bool(end),expected_count=len(masks[agent]))
                if not all(np.isfinite(item).all() for item in row.values()):raise ValueError('Nonfinite multi-agent transition')
                rows[agent].append(row);transitions+=1
                if end or cut:states.pop(agent,None)
            observations={agent:value for agent,value in future.items() if not terminated.get(agent,False) and not truncated.get(agent,False)}
            current_infos=infos
    return {agent:{key:np.asarray([row[key] for row in values]) for key in values[0]} for agent,values in rows.items() if values}


def multiagent_fit(datasets,*,output_dir='./nil_multiagent',algorithm='behavior_cloning',per_agent_options=None,**options):
    from nil.rl._offline import fit
    if not isinstance(datasets,dict) or not datasets: raise ValueError('Provide agent-keyed transition datasets')
    per_agent_options=per_agent_options or {}
    if set(per_agent_options)-set(datasets): raise ValueError('Unknown per-agent configuration')
    results={}
    for index,(agent,data) in enumerate(datasets.items()):
        selected={key:data[key] for key in ('observations','actions')} if algorithm=='behavior_cloning' else data
        results[agent]=fit(selected,algorithm=algorithm,output_dir=Path(output_dir)/f'agent-{index}',**{**options,**per_agent_options.get(agent,{})})
    return results
