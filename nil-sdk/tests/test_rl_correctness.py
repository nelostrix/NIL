"""Analytic rollout boundaries, replay continuation and honest agent semantics."""

import json
import random

import pytest

from nil.rl import AdvantageEstimator, ReplayBuffer, Transition, ValueFunction, RLConfig, RLEnabledAgent


def test_gae_never_leaks_across_terminal_or_truncated_episodes():
    estimator = AdvantageEstimator()
    assert estimator.compute_gae([1., 2., 100.], [.5, 1., 10.], gamma=.9, lam=.8,
        terminated=[False, True, True], next_values=[1., 0., 0.]) == pytest.approx([2.12, 1., 90.])
    assert estimator.compute_gae([1., 100.], [.5, 0.], gamma=.9, lam=.8,
        terminated=[False, True], truncated=[True, False], next_values=[5., 0.]) == pytest.approx([5., 100.])
    assert estimator.compute_monte_carlo([1., 2., 100.], gamma=.9,
        terminated=[False, True, True]) == pytest.approx([2.8, 2., 100.])
    assert estimator.compute_monte_carlo([1., 100.], gamma=.9,
        terminated=[False, True], truncated=[True, False], next_values=[5., 0.]) == pytest.approx([5.5, 100.])
    with pytest.raises(ValueError, match='final observations'):
        estimator.compute_gae([1.], [0.], truncated=[True])
    with pytest.raises(ValueError, match='align'):
        estimator.compute_gae([1.], [])


def test_bounded_ring_replay_is_ordered_and_resume_preserves_rng():
    random.seed(17)
    global_rng = random.getstate()
    replay = ReplayBuffer(5, seed=3)
    for index in range(10):
        replay.push(Transition(state={'index': index}, reward=float(index), log_prob=-.2,
                               truncated=index == 9))
    assert [item.state['index'] for item in replay.latest(3)] == [7, 8, 9]
    assert replay.latest(0) == []
    restored = ReplayBuffer(5).load_state_dict(json.loads(json.dumps(replay.state_dict())))
    assert [item.state for item in replay.sample(3)] == [item.state for item in restored.sample(3)]
    assert restored.latest(1)[0].truncated and not restored.latest(1)[0].terminated
    assert restored.to_dict_list()[-1]['log_prob'] == -.2
    assert random.getstate() == global_rng
    with pytest.raises(ValueError):
        replay.sample(-1)
    with pytest.raises(ValueError):
        ReplayBuffer(0)


def test_value_gradient_uses_preupdate_hidden_weights():
    value = ValueFunction(state_dim=1, hidden_dim=1)
    loss = value.update([{'service_health': 'healthy'}], [2.])
    assert loss == pytest.approx((.01 - 2.) ** 2)
    assert value._hidden_weights[0] == pytest.approx(.10398)
    assert value._weights[0][0] == pytest.approx(.10398)
    with pytest.raises(ValueError, match='align'):
        value.update([{}], [])


def test_agent_updates_use_ordered_rollouts_and_do_not_claim_ppo(monkeypatch):
    assert RLConfig().algorithm == 'advantage_weights'
    with pytest.raises(ValueError, match='neural PPO'):
        RLConfig(algorithm='ppo')
    agent = RLEnabledAgent()
    agent.online_rl_config.batch_size = 3
    for index in range(5):
        agent.replay_buffer.push(Transition(action='search', reward=float(index), done=index == 4))
    seen = []
    original = agent.advantage_estimator.compute_gae

    def record(rewards, *args, **kwargs):
        seen.extend(rewards)
        return original(rewards, *args, **kwargs)

    monkeypatch.setattr(agent.advantage_estimator, 'compute_gae', record)
    agent._update_policy()
    assert seen == [2., 3., 4.]
