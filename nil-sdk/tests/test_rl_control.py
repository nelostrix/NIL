"""Actual SB3 neural learning, bounded evaluation and complete native bundles."""

import copy
import random
import threading

import pytest

sb3 = pytest.importorskip('stable_baselines3')
import gymnasium as gym
import numpy as np
import torch

import nil

torch.set_num_threads(1)


class Bandit(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, np.float32), {}

    def step(self, action):
        return np.zeros(1, np.float32), float(action == 1), True, False, {}


class ContinuousBandit(Bandit):
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def step(self, action):
        reward = -float((np.asarray(action).reshape(-1)[0] - .5) ** 2)
        return np.zeros(1, np.float32), reward, True, False, {}


def small_options(name):
    options = {'policy_kwargs': {'net_arch': [16, 16]}}
    if name == 'ppo':
        options.update(n_steps=16, batch_size=16, n_epochs=2)
    elif name == 'a2c':
        options.update(n_steps=8)
    else:
        options.update(buffer_size=64, batch_size=8, learning_starts=4, train_freq=1, gradient_steps=1)
    return options


@pytest.mark.parametrize('name', ['ppo', 'a2c', 'dqn', 'sac', 'td3', 'ddpg'])
def test_each_native_algorithm_trains_and_roundtrips(tmp_path, name):
    env = Bandit() if name in {'ppo', 'a2c', 'dqn'} else ContinuousBandit()
    model = nil.rl.create(name, env, seed=7, **small_options(name))
    assert type(model).__name__ == name.upper()
    before = copy.deepcopy(model.policy.state_dict())
    result = nil.rl.train(model, total_timesteps=32, output_dir=tmp_path / 'runs')
    assert result.status == 'completed' and result.actual_steps == 32
    assert any(not torch.equal(value, before[key]) for key, value in model.policy.state_dict().items())
    observation = np.zeros(1, np.float32)
    expected = model.predict(observation, deterministic=True)[0]
    np.testing.assert_array_equal(nil.predict(result, observation, deterministic=True)[0], expected)
    evaluation_env = type(env)()
    assert nil.evaluate(result, evaluation_env, episodes=2, max_steps=1)['total_steps'] == 2
    evaluation_env.close()
    path = result.save(tmp_path / name)
    with pytest.raises(ValueError, match='trusted=True'):
        nil.rl.load(path)
    restored = nil.rl.load(path, env=type(env)(), trusted=True, restore_rng=True)
    np.testing.assert_array_equal(restored.predict(observation, deterministic=True)[0], expected)
    assert restored.num_timesteps == model.num_timesteps
    if name not in {'ppo', 'a2c'}:
        assert restored.replay_buffer.size() == model.replay_buffer.size()
    continued = nil.rl.train(restored, total_timesteps=16, output_dir=tmp_path / 'continued')
    assert continued.actual_steps == 16 and restored.num_timesteps == 48
    with pytest.raises(FileExistsError):
        nil.rl.save(restored, path)
    with (path / 'model.zip').open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        nil.rl.load(path, trusted=True)
    model.get_env().close()
    restored.get_env().close()


def test_ppo_and_dqn_actually_learn_reward_preference(tmp_path):
    for name in ('ppo', 'dqn'):
        options = small_options(name)
        options['learning_rate'] = .01
        model = nil.rl.create(name, Bandit(), seed=3, **options)
        nil.rl.train(model, total_timesteps=512, output_dir=tmp_path / name)
        measured = nil.rl.evaluate(model, Bandit(), episodes=20, max_steps=1)
        assert measured['mean_return'] > .9
        model.get_env().close()


def test_sac_improves_continuous_action_reward(tmp_path):
    options = small_options('sac')
    options.update(learning_rate=.003, buffer_size=512, batch_size=32, ent_coef=.01)
    model = nil.rl.create('sac', ContinuousBandit(), seed=3, **options)
    before = nil.rl.evaluate(model, ContinuousBandit(), episodes=5, max_steps=1)['mean_return']
    nil.rl.train(model, total_timesteps=512, output_dir=tmp_path)
    after = nil.rl.evaluate(model, ContinuousBandit(), episodes=5, max_steps=1)['mean_return']
    assert after > before + .02
    model.get_env().close()


def test_evaluation_isolated_rng_normalization_and_step_bounds(tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    train_env = gym.make('CartPole-v1')
    vector = VecNormalize(DummyVecEnv([lambda: train_env]))
    model = nil.rl.create('ppo', vector, **small_options('ppo'))
    nil.rl.train(model, total_timesteps=32, output_dir=tmp_path / 'runs')
    mean, count = vector.obs_rms.mean.copy(), vector.obs_rms.count
    torch_rng, numpy_rng, python_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    observation = model._last_obs.copy()
    mode = model.policy.training
    report = nil.rl.evaluate(model, gym.make('CartPole-v1'), episodes=2, max_steps=3, deterministic=False)
    assert report['total_steps'] <= 6
    np.testing.assert_array_equal(vector.obs_rms.mean, mean)
    assert vector.obs_rms.count == count
    np.testing.assert_array_equal(model._last_obs, observation)
    torch.testing.assert_close(torch_rng, torch.get_rng_state(), rtol=0, atol=0)
    assert python_rng == random.getstate()
    for first, second in zip(numpy_rng, np.random.get_state()):
        np.testing.assert_equal(first, second)
    assert model.policy.training == mode
    with pytest.raises(ValueError, match='separate'):
        nil.rl.evaluate(model, train_env)
    path = nil.rl.save(model, tmp_path / 'normalized')
    restored = nil.rl.load(path, env=gym.make('CartPole-v1'), trusted=True)
    np.testing.assert_array_equal(restored.get_vec_normalize_env().obs_rms.mean, mean)
    assert restored.get_vec_normalize_env().obs_rms.count == count
    model.get_env().close()
    restored.get_env().close()


def test_cancellation_replay_preflight_and_collection_rounding(tmp_path):
    with pytest.raises(ValueError, match='max_replay_bytes'):
        nil.rl.create('dqn', Bandit(), buffer_size=10000, max_replay_bytes=1)
    model = nil.rl.create('ppo', Bandit(), **small_options('ppo'))
    cancelled = threading.Event()
    cancelled.set()
    result = nil.rl.train(model, total_timesteps=32, cancel_event=cancelled, output_dir=tmp_path)
    assert result.status == 'cancelled' and result.actual_steps <= 1
    model.get_env().close()
    options = small_options('dqn')
    options['train_freq'] = 4
    model = nil.rl.create('dqn', Bandit(), **options)
    result = nil.rl.train(model, total_timesteps=21, output_dir=tmp_path)
    assert result.actual_steps == 24
    model.get_env().close()
    # NIL defaults must also work for algorithms whose upstream collection
    # default is episode-based and therefore not step bounded.
    model = nil.rl.create('ddpg', ContinuousBandit(), learning_starts=4, buffer_size=64,
                          batch_size=8, policy_kwargs={'net_arch': [16, 16]})
    assert nil.rl.train(model, total_timesteps=8, output_dir=tmp_path).actual_steps == 8
    model.get_env().close()


class GoalBandit(gym.Env):
    observation_space = gym.spaces.Dict({name: gym.spaces.Box(-2, 2, (1,), dtype=np.float32)
        for name in ('observation', 'achieved_goal', 'desired_goal')})
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position, self.steps = 0., 0
        return self.observation(), {}

    def observation(self):
        return {'observation': np.array([self.position], np.float32),
                'achieved_goal': np.array([self.position], np.float32),
                'desired_goal': np.ones(1, np.float32)}

    def compute_reward(self, achieved_goal, desired_goal, info):
        return -np.sum((np.asarray(achieved_goal) - np.asarray(desired_goal)) ** 2, axis=-1)

    def step(self, action):
        self.position = float(np.clip(self.position + float(action[0]) * .1, -2, 2))
        self.steps += 1
        observation = self.observation()
        return observation, float(self.compute_reward(observation['achieved_goal'], observation['desired_goal'], {})), False, self.steps >= 4, {}


def test_her_replay_and_goal_environment_roundtrip(tmp_path):
    options = small_options('sac')
    options.update(learning_starts=8, replay_buffer_class=nil.rl.her_replay_buffer(),
                    replay_buffer_kwargs={'n_sampled_goal': 2, 'goal_selection_strategy': 'future'})
    model = nil.rl.create('sac', GoalBandit(), policy='MultiInputPolicy', **options)
    nil.rl.train(model, total_timesteps=32, output_dir=tmp_path / 'runs')
    path = nil.rl.save(model, tmp_path / 'her')
    restored = nil.rl.load(path, env=GoalBandit(), trusted=True)
    assert isinstance(restored.replay_buffer, nil.rl.her_replay_buffer())
    assert restored.replay_buffer.size() == model.replay_buffer.size()
    observation = GoalBandit().reset(seed=0)[0]
    np.testing.assert_array_equal(model.predict(observation, deterministic=True)[0], restored.predict(observation, deterministic=True)[0])
    nil.rl.train(restored, total_timesteps=16, output_dir=tmp_path / 'continued')
    model.get_env().close()
    restored.get_env().close()


def test_owned_control_worker_returns_trusted_bundle_metadata(tmp_path):
    import os
    if os.name != 'posix':
        pytest.skip('Owned worker acceptance requires POSIX')
    job = nil.rl.submit('ppo', 'CartPole-v1', total_timesteps=32,
        n_steps=16, batch_size=16, n_epochs=1, policy_kwargs={'net_arch': [16, 16]},
        limits=nil.execution.Limits(timeout=30), output_dir=tmp_path / 'runs', job_root=tmp_path / 'jobs')
    result = job.result()
    assert result['status'] == 'completed' and result['actual_steps'] == 32
    restored = nil.rl.load(result['bundle'], env=nil.rl.make_env('CartPole-v1'), trusted=True)
    assert restored.num_timesteps == 32
    restored.get_env().close()
