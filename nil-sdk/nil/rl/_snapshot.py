"""Explicit, narrow exact simulator continuation. No arbitrary wrapper guessing."""
import copy
from nil.runtime import require

_MONITOR=('rewards','needs_reset','episode_returns','episode_lengths','episode_times','total_steps','current_reset_info')
_VECTOR=('buf_obs','buf_dones','buf_rews','buf_infos','reset_infos','_seeds','_options')
_MODEL=('_last_obs','_last_episode_starts','_last_original_obs')


def _parts(model):
    sb3=require('stable_baselines3','rl'); gym=require('gymnasium','rl')
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    if type(model) is not sb3.DQN or model.device.type!='cpu' or type(model.get_env()) is not DummyVecEnv or model.n_envs!=1:
        raise ValueError('Exact simulator continuation supports CPU DQN with one DummyVecEnv only')
    vector=model.get_env(); monitor=vector.envs[0]
    if type(monitor) is not Monitor or monitor.results_writer is not None:
        raise ValueError('Exact simulator continuation requires the standard file-free Monitor')
    env=monitor.env
    if isinstance(env,gym.Wrapper) or any(not callable(getattr(env,key,None)) for key in ('state_dict','validate_state_dict','load_state_dict')):
        raise ValueError('Raw environment must implement state_dict/validate_state_dict/load_state_dict; wrappers unsupported')
    from stable_baselines3.common.buffers import ReplayBuffer
    if type(model.replay_buffer) is not ReplayBuffer: raise ValueError('Exact simulator continuation requires ordinary replay')
    return vector,monitor,env


def capture(model):
    if not getattr(model,'_nil_safe_boundary',False): raise ValueError('Save at a completed NIL training/cancellation boundary')
    vector,monitor,env=_parts(model)
    state=copy.deepcopy(env.state_dict()); env.validate_state_dict(copy.deepcopy(state))
    spaces={key:copy.deepcopy(getattr(env,key).np_random.bit_generator.state) for key in ('action_space','observation_space')}
    return copy.deepcopy({'schema':1,'environment':type(env).__module__+'.'+type(env).__qualname__,
        'state':state,'spaces':spaces,'env_rng':env.np_random.bit_generator.state,
        'monitor':{key:getattr(monitor,key) for key in _MONITOR},'vector':{key:getattr(vector,key) for key in _VECTOR},
        'model':{key:getattr(model,key) for key in _MODEL if hasattr(model,key)}})


def restore(model,snapshot):
    vector,monitor,env=_parts(model)
    if not isinstance(snapshot,dict) or snapshot.get('schema')!=1 or snapshot.get('environment')!=type(env).__module__+'.'+type(env).__qualname__:
        raise ValueError('Environment snapshot type/schema mismatch')
    if set(snapshot.get('monitor',{}))!=set(_MONITOR) or set(snapshot.get('vector',{}))!=set(_VECTOR) or '_last_obs' not in snapshot.get('model',{}) or set(snapshot['model'])-set(_MODEL):
        raise ValueError('Incomplete environment snapshot')
    env.validate_state_dict(copy.deepcopy(snapshot['state']))  # protocol: validation must not mutate
    # Preflight generator states before modifying the supplied environment.
    np=require('numpy','rl')
    for state in [snapshot['env_rng'],*snapshot['spaces'].values()]:
        generator=np.random.default_rng(); generator.bit_generator.state=copy.deepcopy(state)
    env.load_state_dict(copy.deepcopy(snapshot['state']))
    env.np_random.bit_generator.state=copy.deepcopy(snapshot['env_rng'])
    for key,state in snapshot['spaces'].items(): getattr(env,key).np_random.bit_generator.state=copy.deepcopy(state)
    for target,key in ((monitor,'monitor'),(vector,'vector'),(model,'model')):
        for name,value in snapshot[key].items(): setattr(target,name,copy.deepcopy(value))
    model._nil_safe_boundary=True
