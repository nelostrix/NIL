# Deep learning and reinforcement learning

The live [TODO checklist](project/TODO.md) defines completion and records validation
for each item. This guide describes the implemented subset. A single NIL API
surface does not mean independent numerical engines: torch and SB3 remain
explicit optional backends. Installing the base wheel still installs NIL only.

## Deep learning

`nil.nn` exposes native torch.nn modules and a small MLP constructor. `nil.train`
and Trainer handle real model/optimizer/scheduler state, clipping, callbacks,
weighted gradient accumulation and epoch-boundary resume. Standard tasks are
regression, multiclass classification, binary_classification,
multilabel_classification, imitation_learning and causal_lm; custom callable
steps cover other objectives. Native convolutional segmentation uses integer
class masks with -100 ignored labels. Binary/multilabel tasks consume logits
and matching-shape targets. Custom steps own their normalization/state contracts.

```python
import nil
import torch

X = torch.ones(8, 2)
data = nil.data.tensors(X, torch.ones(8, 1))
model = nil.nn.mlp(2, 1, hidden=(8,))
result = nil.train(model, data, epochs=3, batch_size=3, accumulation_steps=2,
                   precision='bf16', device='cpu')
print(nil.evaluate(result.model, data, device_name='cpu'))
```

CPU bf16 autocast training and dropout/shuffle resume are tested. CUDA fp16
uses native GradScaler with saved state, unscale-before-clip and skipped unsafe
updates; actual CUDA validation remains open. fp32 remains the default and
bf16 may be slower on older CPUs. CNN/LSTM/Transformer examples in
`nil-sdk/examples/deep_workflows.py` execute forward/backward, tiny learning
checks and exact state-dict roundtrips without downloading weights.

Checkpoints validate structure/shapes/dtypes and standard optimizer/scheduler
classes before state mutation. Incompatible resume configuration/precision is
rejected before loading weights/RNG. Generational publication retains one prior
checkpoint and the regular .pt alias. `nil.training.checkpoint.recover(path)`
repairs an interrupted publication from its committed manifest;
`recover(path, previous=True)` explicitly rolls back one generation. This needs
one writer per path and extra disk space for the retained generation. Legacy
valid version-1 files still load. Full custom-loader, callback, augmentation,
cross-device and distributed continuation requires further contracts/tests.

## Neural reinforcement learning

`nil.rl.create` provides original SB3 PPO/A2C/DQN/SAC/TD3/DDPG objects. HER is
an explicit replay-buffer option for a compatible goal environment. CPU is the
default, no vector worker/rendering service starts automatically, and off-policy
replay defaults to 10000 rows with a raw-array allocation estimate. These are
integrations, not NIL-native algorithm reimplementations.

```python
import nil

model = nil.rl.create('ppo', 'CartPole-v1', seed=7, n_steps=64, batch_size=32)
result = nil.rl.train(model, total_timesteps=512)
env = nil.rl.make_env('CartPole-v1')
print(nil.evaluate(result, env, episodes=2, max_steps=200))
result.save('./policy-v1')
restored = nil.rl.load('./policy-v1', env=nil.rl.make_env('CartPole-v1'), trusted=True)
model.get_env().close()
restored.get_env().close()
env.close()
```

Training reports requested and actual transitions; native collection rounds up
to a rollout or step-based train_freq chunk. Cancellation is cooperative through
an optional Event; hard POSIX worker timeout/cancellation is separate. Episode-
based collection is rejected by the bounded helper. Evaluation is limited by
episodes and per-episode max_steps and uses a separate raw environment. Existing
observation normalization is applied without updating training statistics;
sequential evaluation restores RNG and policy mode. Arbitrary concurrent use of
global backend RNG is not isolated. Returned rewards are raw environment returns.

RL bundles retain native policy/optimizer, optional replay/VecNormalize state
and RNG snapshots with version/hash checks. They require trusted=True because
SB3/Python metadata can execute code. Fresh attached environments are reset;
arbitrary simulator state is not saved, and partial replay trajectories may be
truncated. Restoring these components is not exact mid-episode continuation.

For hard process timeout/cancellation, `nil.rl.submit` constructs the algorithm
from its name and a registered Gymnasium environment ID inside an owned worker,
then returns plain bundle/run metadata. Use `job.result()` and explicitly load
the returned trusted bundle. Live SB3 algorithms contain schedules/environments
that standard pickle cannot generally transfer through generic workers.

Legacy RLEnabledAgent/Policy/BehavioralCloning/DirectPreferenceOptimizer are agent
action-weight heuristics. They are not neural PPO, offline CQL, neural BC or full
RLHF/DPO. Their terminal/truncation-aware return helpers require ordered rollouts
and final-observation values for truncation bootstrap. Replay uses O(1) ring
insertion and local RNG serialization. Misleading agent algorithm='ppo' is
rejected; use the neural control API above.

## Validation and remaining gates

```sh
python scripts/verify_neural.py --rl
python scripts/verify_neural.py --rl --cuda  # fails preflight without real GPU
python nil-sdk/examples/deep_workflows.py --architecture transformer
python nil-sdk/examples/rl_workflow.py --algorithm ppo
```

Run verification commands from the source checkout; extracted distribution
bundles provide runnable examples and the standalone core-install verifier.
The tests include native loss/gradient/metric parity, actual checkpoint-writer
interruption, CPU AMP resume, all six neural control algorithm updates/roundtrips,
HER, replay/normalization continuation, bounded evaluation, RNG isolation and
tiny discrete/continuous learning fixtures. They are not broad multi-seed policy
benchmarks. DDP/FSDP, offline/recurrent/masked/multi-agent RL, custom simulator
continuation, GPU/hardware matrices and independent engines remain open in TODO.md.


### Validated loader/stream continuation

Training now exposes bounded worker/prefetch controls (default zero workers),
optional pinning, native sampler epoch hooks, loader/sampler generator state,
and explicit validated dataset/sampler state. `nil.data.jsonl_stream` supports
single-worker, immutable-source cursor continuation in bounded chunks. Resume
preflights descriptor/cursor contracts before model mutation. Persistent workers,
unacknowledged iterable prefetch and unbounded iterables are explicitly rejected.
See `nil.help('training')` and `nil.help('data')` for scope and examples. Arbitrary
map data/transform/collate content is caller-owned, not fingerprinted by type/length.

## Extended CPU RL contracts — 2026-11-23

`collect_multiagent(..., masked_agents=('agent',))` supports explicit masked agents
and automatically detects native MaskablePPO. Masks come from wrapped observations
`{'observation': array, 'action_mask': mask}`, info or env.action_masks(agent).
Native recurrent policies carry state; compatible custom predictors can declare
nil_recurrent=True and accept state/episode_start. Native RecurrentPPO does not
accept masks and is refused for a combined masked contract. Per-agent Box/Discrete
spaces, when provided, validate observations/actions independently; heterogeneous
agents remain separate arrays. All masks/actions/terminal flags are checked.

Terminal observations in info take precedence over reset-looking future values.
Truncation requires a real terminal observation and valid terminal_action_masks;
true termination can use a zero next mask because it never bootstraps. Agent state
is discarded on end and starts fresh on reappearance/reset. RNG/mode preservation,
transition limits and finite data checks apply to successful and failed collection.

`offline_fit(..., algorithm='cql', action_count=..., conservative_weight=1.)`
adds a discrete conservative fitted-Q objective: ordinary terminal-aware target
loss plus alpha*(logsumexp(valid Q)-Q(recorded action)). Only valid current actions
enter the regularizer. Finite next-action masks are required for nonterminal and
truncated rows. The existing bounded transition schema, target network, exact
same-data/config resume and trusted policy bundles apply. This is a discrete
CQL-style baseline, not continuous-action CQL, IQL or a guarantee for arbitrary
out-of-distribution datasets. No new package is required beyond explicit Torch.

`nil.rl.centralized_fit(datasets, action_counts={...}, returns=team_returns)` is
an offline centralized-value advantage-weighted regression algorithm. Datasets
contain aligned per-agent observations/actions and optional action_masks; returns
contain one explicit discounted team return per aligned row. A critic sees joint
observations; separate actors see only their own observations during inference.
Weights are exp((return-value)/temperature), bounded by max_weight. It supports
1–16 fixed discrete agents, flat finite observations, bounded models/datasets,
local deterministic sampling and same-data/config optimizer resume. It does not
infer row alignment or simulator returns, collect online exploration, handle
variable populations, or implement MAPPO. Call result.policies[agent].predict(...)
for decentralized inference. Reload its checkpoint with
centralized_load(path, trusted=True). Checkpoint restore is for weights/training;
it does not resume an environment or later external actor sampling history.

Exact simulator continuation additionally supports CPU PPO/A2C with ordinary
RolloutBuffer, flat Box observations, discrete actions and no SDE at a completed
NIL rollout/update boundary. The raw environment still needs the explicit
state_dict/validate_state_dict/load_state_dict protocol, one DummyVecEnv, a standard
file-free Monitor and identical backend versions. No arbitrary-wrapper, recurrent,
continuous-action or multi-environment exact-state guarantee is implied. DQN still
requires ordinary replay; PPO/A2C have no replay archive to restore. For all three,
compare identical learn-call chunk schedules and explicitly restore RNG/state.

## Measured broader learning scope

[Raw three-seed CPU evidence](benchmarks/2026-11-23-rl-coverage.json) includes:

| Contract | Task/data | Learning result |
| --- | --- | --- |
| Discrete CQL | Fixed deterministic FrozenLake4x4, 1,000 random and256 scripted-expert replay episodes | All learned policies reached the goal; paired mean improvement0.95 over random |
| Centralized-value AWR | Two real CartPole environments as a fixed cooperative team;32 expert joint episodes, explicit discounted team returns | Paired mean improvement484.42 joint steps over random |
| Continuous BC | MountainCarContinuous,64 velocity-controller expert episodes, explicit observation scaling | Paired mean improvement122.98 return over random |
| Masked PPO | Three-action contextual task with one forbidden distractor | Paired mean improvement0.49 over random-valid actions |
| Recurrent PPO | Fully observed CartPole,8,192 training steps/seed | Paired mean improvement223.9 steps over random |

Seeds3/7/11 are the statistical sampling units; evaluation episodes are held out.
The report contains per-seed returns and bootstrap intervals. FrozenLake uses the
same map/dynamics; the masked task is one-step; recurrent CartPole does not test
partial-observation memory learning. Offline expert replay is explicit, not an
online-exploration claim. These tests establish declared task learning, not general
algorithm superiority. Reproduce with scripts/benchmark_rl_coverage.py; its cases
can be selected independently. No models or datasets are downloaded by the runner.
