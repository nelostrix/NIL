"""Optional native SB3 integration; no backend loads during module import."""

from dataclasses import dataclass
from importlib import metadata
import json
import math
from pathlib import Path
import pickle
import platform
import random
import shutil
import tempfile
import uuid

from nil._classical_common import positive_int
from nil.lifecycle import Run, sha256
from nil.runtime import require

_ALGORITHMS = {'ppo': 'PPO', 'a2c': 'A2C', 'dqn': 'DQN', 'sac': 'SAC', 'td3': 'TD3', 'ddpg': 'DDPG'}


_EXTENDED = {'recurrent_ppo':'RecurrentPPO','maskable_ppo':'MaskablePPO'}

def algorithms(include_contrib=False):
    return sorted({**_ALGORITHMS,**(_EXTENDED if include_contrib else {})})


def make_env(name, **parameters):
    return require('gymnasium', 'rl').make(name, **parameters)


def her_replay_buffer():
    return require('stable_baselines3', 'rl').HerReplayBuffer


def submit(algorithm, env_id, *, total_timesteps, limits=None, job_root='./nil_jobs',
           output_dir='./nil_runs', **model_options):
    """Construct/train/save inside an owned worker; return plain bundle metadata."""
    if not isinstance(env_id, str):
        raise TypeError('Owned RL jobs require a registered Gymnasium environment ID')
    positive_int(total_timesteps, 'total_timesteps')
    if not isinstance(algorithm, str) or algorithm.lower() not in algorithms(include_contrib=True):
        raise ValueError('Choose one of nil.rl.algorithms()')
    from nil.execution import submit as submit_worker
    root = Path(output_dir).resolve()
    artifact = root / ('rl-policy-' + uuid.uuid4().hex)
    return submit_worker('nil.rl._control:_train_job', algorithm, env_id, total_timesteps,
                         str(root), str(artifact), model_options, limits=limits, job_root=job_root)


def _train_job(algorithm, env_id, steps, output_dir, artifact, options):
    model = create(algorithm, env_id, **options)
    try:
        result = train(model, total_timesteps=steps, output_dir=output_dir)
        result.run.event('checkpoint.publication', {'path': artifact})
        bundle = save(model, artifact)
        result.run.event('checkpoint.saved', {'path': str(bundle)})
        return {'status': result.status, 'actual_steps': result.actual_steps,
                'requested_steps': steps, 'bundle': str(bundle), 'run': str(result.run.path)}
    finally:
        model.get_env().close()


def _algorithm(name):
    if not isinstance(name, str) or name.lower() not in algorithms(include_contrib=True):
        raise ValueError('Choose one of nil.rl.algorithms(): ' + ', '.join(algorithms()))
    if name.lower() in _EXTENDED:
        return getattr(require('sb3_contrib','rl-contrib'),_EXTENDED[name.lower()])
    return getattr(require('stable_baselines3', 'rl'), _ALGORITHMS[name.lower()])


def _space_bytes(space):
    if hasattr(space, 'spaces'):
        parts = space.spaces.values() if isinstance(space.spaces, dict) else space.spaces
        return sum(_space_bytes(part) for part in parts)
    np = require('numpy', 'rl')
    return math.prod(space.shape or ()) * np.dtype(space.dtype).itemsize


def create(algorithm, env, *, policy='MlpPolicy', seed=0, device='cpu',
           max_replay_bytes=256 * 1024 ** 2, **parameters):
    """Return an original SB3 algorithm; default off-policy buffers are bounded."""
    cls = _algorithm(algorithm)
    name = algorithm.lower()
    if name=='recurrent_ppo' and policy=='MlpPolicy': policy='MlpLstmPolicy'
    gym = require('gymnasium', 'rl')
    owned = isinstance(env, str)
    env = gym.make(env) if owned else env
    try:
        if isinstance(policy, str) and policy in {'MlpPolicy', 'MultiInputPolicy','MlpLstmPolicy'}:
            policy_options = dict(parameters.get('policy_kwargs', {}))
            policy_options.setdefault('net_arch', [64, 64])
            if name=='recurrent_ppo': policy_options.setdefault('lstm_hidden_size',32)
            parameters['policy_kwargs'] = policy_options
        if name in {'dqn', 'sac', 'td3', 'ddpg'}:
            parameters.setdefault('buffer_size', 10000)
            parameters.setdefault('train_freq', 4 if name == 'dqn' else 1)
            size = positive_int(parameters['buffer_size'], 'buffer_size')
            positive_int(max_replay_bytes, 'max_replay_bytes')
            estimated = size * getattr(env, 'num_envs', 1) * (
                2 * _space_bytes(env.observation_space) + _space_bytes(env.action_space) + 16)
            if estimated > max_replay_bytes:
                raise ValueError('Estimated replay arrays exceed max_replay_bytes; reduce buffer_size')
        if name in {'ppo',*_EXTENDED}:
            parameters.setdefault('n_steps', 128)
            parameters.setdefault('batch_size', 64)
        return cls(policy, env, seed=seed, device=device, **parameters)
    except BaseException:
        if owned:
            env.close()
        raise


@dataclass
class RLTrainingResult:
    model: object
    run: Run
    requested_steps: int
    actual_steps: int
    status: str
    _nil_rl_result = True

    def predict(self, observation, **options):
        return self.model.predict(observation, **options)

    def save(self, path, **options):
        return save(self.model, path, **options)

    def evaluate(self, env, **options):
        return evaluate(self.model, env, **options)


def train(model, *, total_timesteps, output_dir='./nil_runs', cancel_event=None,
          callback=None, reset_num_timesteps=False, **options):
    """Native learning with explicit step accounting and cooperative cancellation."""
    positive_int(total_timesteps, 'total_timesteps')
    model = model.model if getattr(model,'_nil_rl_result',False) else model
    if model.get_env() is None:
        raise ValueError('Training requires an attached environment')
    callbacks = require('stable_baselines3.common.callbacks', 'rl')
    np = require('numpy', 'rl')
    if cancel_event is not None and not callable(getattr(cancel_event, 'is_set', None)):
        raise TypeError('cancel_event must provide is_set()')
    options.setdefault('progress_bar', False)
    on_policy = require('stable_baselines3.common.on_policy_algorithm', 'rl').OnPolicyAlgorithm
    chunk = model.n_steps if isinstance(model, on_policy) else None
    if chunk is None:
        frequency = getattr(model, 'train_freq', None)
        if frequency is not None and getattr(frequency.unit, 'value', frequency.unit) != 'step':
            raise ValueError('Bounded step training requires step-based train_freq, not episode-based collection')
        chunk = frequency.frequency if frequency is not None else 1
    chunk = positive_int(int(chunk), 'collection_chunk') * model.n_envs
    upper = math.ceil(total_timesteps / chunk) * chunk
    start = 0 if reset_num_timesteps else model.num_timesteps
    run = Run(output_dir, 'rl-' + type(model).__name__.lower(),
              {'requested_steps': total_timesteps, 'collection_step_bound': upper, 'device': str(model.device)})

    class CancelAtBoundary(Exception): pass

    model._nil_safe_boundary = False
    class Guard(callbacks.BaseCallback):
        cancelled = False

        def _on_rollout_start(self):
            if cancel_event is not None and cancel_event.is_set():
                self.cancelled = True
                raise CancelAtBoundary()

        def _on_step(self):
            if not np.isfinite(self.locals.get('rewards', [])).all():
                raise ValueError('Environment returned non-finite rewards')
            if self.model.num_timesteps - start > upper:
                raise RuntimeError('Backend exceeded its declared collection-step bound')
            return True

    guard = Guard()
    supplied = callback if isinstance(callback, list) else [callback] if callback is not None else []
    combined = [guard, *[item if isinstance(item, callbacks.BaseCallback) else callbacks.ConvertCallback(item)
                         for item in supplied]]
    run.status('running')
    try:
        try:
            model.learn(total_timesteps=total_timesteps, callback=combined,
                        reset_num_timesteps=reset_num_timesteps, **options)
        except CancelAtBoundary:
            pass
        actual = model.num_timesteps - start
        model._nil_safe_boundary = guard.cancelled or actual >= total_timesteps
        state = 'cancelled' if guard.cancelled else 'completed'
        run.log({'actual_steps': actual}, actual)
        run.status(state, actual_steps=actual, requested_steps=total_timesteps,
                   reason='callback_or_algorithm_stop' if actual < total_timesteps else 'budget')
        return RLTrainingResult(model, run, total_timesteps, actual, state)
    except BaseException as error:
        run.status('cancelled' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise


def evaluate(model, env, *, episodes=5, max_steps=1000, seed=0, deterministic=True):
    """Evaluate on a separate raw Gymnasium env, without changing training RMS/RNG."""
    positive_int(episodes, 'episodes')
    model = model.model if getattr(model,'_nil_rl_result',False) else model
    positive_int(max_steps, 'max_steps')
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    gym = require('gymnasium', 'rl')
    if not isinstance(env, gym.Env):
        raise TypeError('Evaluation requires a separate raw Gymnasium Env')
    cursor = model.get_env()
    while hasattr(cursor, 'venv'):
        cursor = cursor.venv
    if hasattr(cursor, 'envs') and any(item.unwrapped is env.unwrapped for item in cursor.envs):
        raise ValueError('Use a separate evaluation environment; never reset a training environment')
    normalization = model.get_vec_normalize_env()
    np = require('numpy', 'rl')
    torch = require('torch', 'rl')
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = [model.device.index if model.device.index is not None else torch.cuda.current_device()] if model.device.type == 'cuda' else []
    records = []
    previous_mode = model.policy.training
    local_rng=model.action_generator.get_state() if hasattr(model,'action_generator') else None
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            for episode in range(episodes):
                observation, _ = env.reset(seed=seed + episode)
                recurrent_state=None; episode_start=True
                reward_sum = 0.0
                terminated = truncated = False
                for step in range(max_steps):
                    chosen = normalization.normalize_obs(observation) if normalization is not None else observation
                    prediction_options={}
                    if type(model).__name__=='RecurrentPPO': prediction_options={'state':recurrent_state,'episode_start':np.asarray([episode_start])}
                    if type(model).__name__=='MaskablePPO':
                        from nil.rl._rollouts import action_mask
                        prediction_options['action_masks']=action_mask(env)
                    action, recurrent_state = model.predict(chosen, deterministic=deterministic,**prediction_options)
                    episode_start=False
                    observation, reward, terminated, truncated, _ = env.step(action)
                    if not math.isfinite(reward):
                        raise ValueError('Evaluation returned a non-finite reward')
                    reward_sum += float(reward)
                    if terminated or truncated:
                        break
                records.append({'return': reward_sum, 'steps': step + 1,
                                'terminated': bool(terminated), 'truncated': bool(truncated),
                                'budget_truncated': not (terminated or truncated)})
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        model.policy.set_training_mode(previous_mode)
        if local_rng is not None: model.action_generator.set_state(local_rng)
    returns = [item['return'] for item in records]
    return {'mean_return': float(np.mean(returns)), 'std_return': float(np.std(returns)),
            'episodes': records, 'total_steps': sum(item['steps'] for item in records)}


def save(model, path, *, include_replay=True, environment_state=False):
    """Publish a new trusted SB3 bundle; never overwrite an existing checkpoint."""
    model = model.model if getattr(model,'_nil_rl_result',False) else model
    if getattr(model,'_nil_rl_policy',False):
        if environment_state: raise ValueError('Offline policies have no simulator state')
        from nil.rl._offline import save as offline_save
        return offline_save(model,path)
    name = next((key for key,value in _EXTENDED.items() if type(model).__name__==value),type(model).__name__.lower())
    if name not in algorithms(include_contrib=True) or not isinstance(model, _algorithm(name)):
        raise TypeError('Only supported native SB3 algorithms can use this bundle format')
    snapshot = None
    if environment_state:
        if not include_replay and getattr(model,'replay_buffer',None) is not None: raise ValueError('Environment continuation requires replay')
        from nil.rl._snapshot import capture
        snapshot = capture(model)
    destination = Path(path).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.nil-rl-', dir=destination.parent))
    try:
        model.save(temporary / 'model.zip')
        if snapshot is not None:
            with (temporary / 'environment.pkl').open('wb') as stream: pickle.dump(snapshot,stream,protocol=5)
        if include_replay and getattr(model, 'replay_buffer', None) is not None:
            model.save_replay_buffer(temporary / 'replay.pkl')
        normalization = model.get_vec_normalize_env()
        if normalization is not None:
            normalization.save(temporary / 'normalization.pkl')
        np, torch = require('numpy', 'rl'), require('torch', 'rl')
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
        if model.device.type == 'cuda':
            index = model.device.index if model.device.index is not None else torch.cuda.current_device()
            rng['cuda'] = {index: torch.cuda.get_rng_state(index)}
        with (temporary / 'rng.pkl').open('wb') as stream:
            pickle.dump(rng, stream, protocol=5)
        manifest = {'schema_version': 1, 'format': 'nil-rl-sb3', 'algorithm': name,
                    'python': platform.python_version(), 'timesteps': model.num_timesteps,
                    'dependencies': {package: metadata.version(package) for package in
                                     ('stable-baselines3', 'gymnasium', 'numpy', 'torch')},
                    'files': {part.name: sha256(part) for part in temporary.iterdir()}}
        if name in _EXTENDED: manifest['dependencies']['sb3-contrib']=metadata.version('sb3-contrib')
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        destination.mkdir()
        try:
            for part in temporary.iterdir():
                if part.name != 'manifest.json':
                    part.replace(destination / part.name)
            (temporary / 'manifest.json').replace(destination / 'manifest.json')
        except BaseException:
            shutil.rmtree(destination)
            raise
        return destination
    finally:
        shutil.rmtree(temporary)


def load(path, *, env=None, trusted=False, device='cpu', restore_rng=False, allow_version_mismatch=False, restore_environment=False):
    if restore_environment and (env is None or not restore_rng or allow_version_mismatch):
        raise ValueError('Exact environment continuation requires env, restore_rng=True and matching versions')
    if not trusted:
        raise ValueError('RL bundles contain executable Python metadata; trusted=True is required')
    root = Path(path)
    if root.is_symlink() or (root / 'manifest.json').is_symlink():
        raise ValueError('RL bundles must not use symlinks')
    manifest = json.loads((root / 'manifest.json').read_text())
    if isinstance(manifest,dict) and manifest.get('format')=='nil-rl-offline-v1':
        if restore_environment: raise ValueError('Offline policies have no simulator state')
        from nil.rl._offline import load as offline_load
        return offline_load(path,trusted=trusted)
    if not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int or manifest.get('schema_version') != 1 or manifest.get('format') != 'nil-rl-sb3':
        raise ValueError('Unsupported RL bundle')
    files = manifest.get('files')
    if not isinstance(files, dict) or 'model.zip' not in files or set(files) - {'model.zip', 'replay.pkl', 'normalization.pkl', 'rng.pkl', 'environment.pkl'}:
        raise ValueError('Invalid RL artifact list')
    for name, expected in files.items():
        artifact = root / name
        if artifact.is_symlink() or not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError('RL bundle integrity check failed')
    dependencies = manifest.get('dependencies')
    if not isinstance(manifest.get('python'), str) or not isinstance(dependencies, dict):
        raise ValueError('Malformed RL environment metadata')
    if not allow_version_mismatch:
        if manifest['python'].split('.')[:2] != platform.python_version().split('.')[:2]:
            raise ValueError('RL checkpoint Python version mismatch')
        for package, version in dependencies.items():
            try:
                current = metadata.version(package)
            except metadata.PackageNotFoundError:
                current = None
            if current != version:
                raise ValueError(f'RL backend version mismatch: {package}')
    if restore_environment and ('environment.pkl' not in files or (manifest.get('algorithm')=='dqn' and 'replay.pkl' not in files)):
        raise ValueError('Bundle lacks exact environment/replay state')
    if 'normalization.pkl' in files:
        if env is None:
            raise ValueError('Restoring observation normalization requires a raw environment')
        vector = require('stable_baselines3.common.vec_env', 'rl')
        if isinstance(env, vector.VecNormalize):
            raise ValueError('Pass an unnormalized environment when restoring normalization')
        if not isinstance(env, vector.VecEnv):
            env = vector.DummyVecEnv([lambda env=env: env])
        env = vector.VecNormalize.load(root / 'normalization.pkl', env)
    model = _algorithm(manifest['algorithm']).load(root / 'model.zip', env=env, device=device)
    if 'replay.pkl' in files:
        model.load_replay_buffer(root / 'replay.pkl')
    if restore_environment:
        from nil.rl._snapshot import restore
        with (root / 'environment.pkl').open('rb') as stream: snapshot=pickle.load(stream)
        restore(model,snapshot)
    if restore_rng:
        with (root / 'rng.pkl').open('rb') as stream:
            rng = pickle.load(stream)
        torch, np = require('torch', 'rl'), require('numpy', 'rl')
        if rng.get('cuda') and (not torch.cuda.is_available() or any(index >= torch.cuda.device_count() for index in rng['cuda'])):
            raise ValueError('Saved CUDA RNG requires matching available devices')
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        for index, state in rng.get('cuda', {}).items():
            torch.cuda.set_rng_state(state, index)
    return model
