# rl

Neural environment training through an optional SB3 backend, plus separate
stdlib agent reward, replay and action-weight helpers.

## Usage

```python
import nil

model = nil.rl.create('ppo', 'CartPole-v1', seed=7, n_steps=64, batch_size=32)
result = nil.rl.train(model, total_timesteps=1024)
evaluation_env = nil.rl.make_env('CartPole-v1')
print(nil.rl.evaluate(model, evaluation_env, episodes=3, max_steps=500))
result.save('./ppo-v1')
restored = nil.rl.load('./ppo-v1', env=nil.rl.make_env('CartPole-v1'), trusted=True)
model.get_env().close()
restored.get_env().close()
evaluation_env.close()
```

## Parameters

`algorithms()` lists ppo/a2c/dqn/sac/td3/ddpg without loading a backend.
`create(name, env, policy='MlpPolicy', seed=0, device='cpu', **native_options)`
returns the actual SB3 algorithm. Off-policy buffer_size defaults to 10000, with
a conservative raw-array estimate capped by max_replay_bytes (256 MiB default).
MLP/multi-input policies default to two 64-unit layers. Off-policy collection
defaults to step-based train_freq (four for DQN, one for SAC/TD3/DDPG).
`her_replay_buffer()` supplies the native HER class for replay_buffer_class;
use MultiInputPolicy and a valid goal environment with compute_reward.
`train` accepts positive total_timesteps, optional cancellation Event and native
callbacks; evaluation has explicit episodes/max_steps and a seed. Save/load
preserve policy/optimizer state, optional replay, VecNormalize statistics and
RNG snapshots. restore_rng=True explicitly restores stored RNG state.

## Output

Native algorithms; RLTrainingResult with model/run/status and requested/actual
step counts. Evaluation reports raw episode returns, lengths and separate
termination, truncation and evaluation-budget flags. Save creates a new immutable
versioned, integrity-checked bundle. Load validates dependencies before restoring.
`nil.predict(model_or_result, observation, deterministic=True)` preserves the
native `(action, recurrent_state)` return. `nil.evaluate(model_or_result,
evaluation_env, episodes=..., max_steps=...)` dispatches to bounded RL evaluation.

## Requirements

`nil-sdk[rl]` for neural training (SB3/Gymnasium/torch). No backend loads when
importing nil.rl, listing algorithms or reading this guide. Agent signal helpers
remain stdlib-only. All integrations are optional, not NIL-native algorithm copies.

## Limitations

Evaluate on a separate raw Gymnasium environment; reusing a training instance is
rejected for locally hosted vector environments. Evaluation applies existing
observation normalization without updating training statistics, and restores
Python/NumPy/torch RNG and policy training mode in sequential use. Shared RNG
isolation is not a concurrent-training guarantee. Close caller-owned environments.
Native learning collects chunks: the reported step bound rounds up to rollout or
step-based train_freq granularity. Episode-based collection is rejected by the
bounded train helper. No wall-time or GPU/RSS cap is implied; nil.execution offers
separate opt-in POSIX worker limits. Replay estimates exclude neural weights,
Python object overhead and workspaces. Rendering and background vector workers
are never started by default. Core algorithms support native action/observation
spaces; recurrent, masked, multi-agent and offline algorithms need further work.

`nil.rl.submit(name, env_id, total_timesteps=..., limits=..., **model_options)`
constructs/trains/saves inside an owned POSIX worker and returns a Job whose
result is plain bundle/run metadata. This avoids pickling live SB3 objects with
their schedules/environments. Load the returned bundle explicitly as trusted.

Bundles contain executable Python metadata; trusted=True is required. Restoring
optimizer/replay/normalization/RNG does not restore arbitrary environment or
simulator state. load attaches a fresh environment and may truncate a partial
replay trajectory; full exact episode continuation is not claimed. GPU behavior
and broad multi-seed learning benchmarks remain unverified here. Tiny regression
environments validate execution and learning paths, not real-world policy quality.

Legacy Policy/BehavioralCloning/DirectPreferenceOptimizer/ValueFunction are agent
action-weight and preference heuristics, not neural PPO, CQL, BC or RLHF/DPO
trainers. RLConfig defaults to advantage_weights; misleading algorithm='ppo'/'cql'
values are rejected. GAE requires an ordered rollout and terminal/truncation
masks; truncation bootstraps require next_values from final observations.
ReplayBuffer uses bounded ring storage, a local seedable RNG, chronological
latest(), and state_dict/load_state_dict for contents/cursor/RNG continuation.
These helpers are not physical robot safety controllers.

## Offline neural learning

```python
import nil
result = nil.rl.offline_fit(
    {'observations': [[-1.], [1.]], 'actions': [0, 1]},
    action_count=2, steps=300, seed=0, output_dir='./offline-run')
print(result.model.predict([[1.]], deterministic=True)[0])
```

`algorithm='behavior_cloning'` uses a neural categorical policy or continuous MSE
policy; 2D continuous action rows distinguish continuous targets. `offline_dqn`
is an explicit fitted-Q baseline, not CQL/IQL. Its dataset also needs
`next_observations`, `rewards`, `terminated`, `truncated`. Optional `action_masks`
and `next_action_masks` are validated. Bootstrap at truncation, never terminal.
Flat finite observations, CPU, bounded dataset (default 64 MB)/network sizes.
`resume=result.checkpoint` restores optimizer, target and local RNG with identical
data/config; `steps` is the final update count. `offline_save`/`offline_load`
provide immutable trusted bundles, also supported by root RL save/load.

## Structured environment contracts

`algorithms(include_contrib=True)` additionally lists `recurrent_ppo` and
`maskable_ppo`; these require the explicit rl-contrib extra. Create/train/save/load
as for PPO. Evaluation carries recurrent state with episode-start resets and
validates masks. `collect_vector(policy, dummy_vec_env, steps=...)` uses separate
CPU evaluation environments, records actual terminal observations through auto
reset, carries recurrent state, and preserves caller RNG/mode. Masked truncated
transitions require `terminal_action_masks` in info.

`collect_multiagent(env, {'agent_id': policy}, steps=...)` accepts a parallel dict
environment returning observation/reward/terminated/truncated/info dictionaries.
`multiagent_fit(datasets, per_agent_options=...)` trains independent neural
policies; this is not centralized-critic MARL. Masked multi-agent collection is
explicitly unsupported, rather than silently dropping terminal masks.

## Exact supported simulator continuation

`save(model, new_path, environment_state=True)` and
`load(path, env=raw_env, trusted=True, restore_rng=True, restore_environment=True)`
restore CPU DQN with ordinary replay, one DummyVecEnv and the standard file-free
Monitor. The raw unwrapped environment must implement `state_dict()`, nonmutating
`validate_state_dict(state)`, and `load_state_dict(state)`. State includes actual
simulator, env/space RNG, vector buffers, monitor counters and last observation.
NIL validates environment identity and exact backend versions. No arbitrary
Gymnasium wrapper/simulator or VecNormalize continuation guarantee is implied.

Cooperative cancellation completes the current collection chunk and its native
update before stopping at the next rollout boundary. A pre-set event stops before
collection. Snapshots require a completed NIL train/cancel boundary; early False
callbacks and failures cannot claim exact continuation. Compare identical learn
chunk schedules: backend schedules may differ from one monolithic learn call.
Measured five-seed learning results and limitations ship in docs/benchmarks.
