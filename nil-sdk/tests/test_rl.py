from __future__ import annotations

import math

import pytest

from nil.rl.agent import RLEnabledAgent, RLConfig
from nil.rl.exploration import ExplorationScheduler
from nil.rl.policy import BehavioralCloning, Policy
from nil.rl.preference import DirectPreferenceOptimizer, EloTrajectoryRating
from nil.rl.replay import ReplayBuffer, Transition
from nil.rl.rewards import (
    DenseRewardModel,
    InverseRLRewardModel,
    RLConfig as RewardConfig,
    ShapedRewardModel,
    StepRecord,
)
from nil.rl.trajectory import TrajectoryScore, TrajectoryScorer
from nil.rl.value import AdvantageEstimator, ValueFunction


# ===========================================================================
# Reward Models
# ===========================================================================

class TestDenseRewardModel:
    def test_step_reward_zero_on_no_change(self):
        model = DenseRewardModel()
        before = {"investigation_stage": "analyzing", "service_health": "degraded", "tools_used": ["log_search"]}
        after = dict(before)
        step = StepRecord(tool_name="log_search")
        r = model.step_reward(before, after, step)
        assert abs(r) < 0.02

    def test_progress_reward_positive(self):
        model = DenseRewardModel()
        before = {"investigation_stage": "not_started", "service_health": "unknown"}
        after = {"investigation_stage": "gathering_data", "service_health": "degraded"}
        r = model._progress_reward(before, after)
        assert r > 0

    def test_progress_reward_negative(self):
        model = DenseRewardModel()
        before = {"investigation_stage": "resolved", "service_health": "healthy"}
        after = {"investigation_stage": "not_started", "service_health": "degraded"}
        r = model._progress_reward(before, after)
        assert r < 0

    def test_cost_penalty(self):
        model = DenseRewardModel()
        step = StepRecord(tool_name="test", tokens_in=500, tokens_out=500)
        penalty = model._cost_penalty(step)
        assert penalty > 0

    def test_error_penalty(self):
        model = DenseRewardModel()
        step = StepRecord(error="fail")
        assert model._error_penalty(step) > 0

    def test_exploration_bonus_for_novel_action(self):
        model = DenseRewardModel()
        state = {"tools_used": ["a", "b"]}
        step = StepRecord(tool_name="c")
        assert model._is_novel_action(state, step) is True
        step2 = StepRecord(tool_name="a")
        assert model._is_novel_action(state, step2) is False

    def test_trajectory_return(self):
        model = DenseRewardModel()
        steps = [
            StepRecord(step_number=1, tool_name="search", tokens_in=50, tokens_out=50),
            StepRecord(step_number=2, tool_name="fix", tokens_in=100, tokens_out=50),
        ]
        states = [
            {"investigation_stage": "not_started", "service_health": "unknown", "uncertainty_level": 1.0, "tools_used": []},
            {"investigation_stage": "gathering_data", "service_health": "degraded", "uncertainty_level": 0.8, "tools_used": ["search"]},
            {"investigation_stage": "resolved", "service_health": "healthy", "uncertainty_level": 0.2, "tools_used": ["search", "fix"]},
        ]
        ret = model.trajectory_return(steps, states)
        assert isinstance(ret, float)


class TestShapedRewardModel:
    def test_potential(self):
        model = ShapedRewardModel()
        state = {"investigation_stage": "diagnosed", "service_health": "degraded", "uncertainty_level": 0.3}
        p = model.potential(state)
        assert p > 0

    def test_shaping_reward(self):
        model = ShapedRewardModel()
        before = {"investigation_stage": "not_started", "service_health": "unknown", "uncertainty_level": 1.0}
        after = {"investigation_stage": "diagnosed", "service_health": "degraded", "uncertainty_level": 0.3}
        shaping = model.shaping_reward(before, after)
        assert shaping != 0

    def test_step_reward_includes_shaping(self):
        model = ShapedRewardModel()
        before = {"investigation_stage": "not_started", "service_health": "unknown", "uncertainty_level": 1.0, "tools_used": []}
        after = {"investigation_stage": "gathering_data", "service_health": "degraded", "uncertainty_level": 0.7, "tools_used": ["s"]}
        r = model.step_reward(before, after, StepRecord(tool_name="s"))
        assert r != 0


class TestInverseRLRewardModel:
    def test_learn_weights(self):
        irl = InverseRLRewardModel(n_features=2)
        trajectories = [
            (
                [{"service_health": "healthy", "investigation_stage": "resolved", "steps_taken": 5, "uncertainty_level": 0.1}],
                [StepRecord(tool_name="fix", step_number=1)],
            ),
        ]
        weights = irl.learn_from_demonstrations(trajectories, n_iterations=10)
        assert len(weights) == 2
        assert all(w >= 0 for w in weights)

    def test_reward_function(self):
        irl = InverseRLRewardModel(n_features=2)
        irl.weights = [2.0, 1.0]
        state = {"service_health": "healthy", "investigation_stage": "resolved", "steps_taken": 5, "uncertainty_level": 0.1}
        r = irl.reward(state, "fix")
        assert r > 0


# ===========================================================================
# Trajectory Scoring
# ===========================================================================

class TestTrajectoryScorer:
    def test_score_successful_trace(self):
        scorer = TrajectoryScorer()
        trace = {
            "outcome_success": True,
            "steps": [
                {"tool_name": "a", "tool_output": "ok"},
                {"tool_name": "b", "tool_output": "ok"},
            ],
            "total_cost": 0.001,
            "total_tokens_in": 50,
            "total_tokens_out": 50,
        }
        score = scorer.score(trace)
        assert score.composite > 0
        assert score.correctness == 1.0

    def test_score_failed_trace(self):
        scorer = TrajectoryScorer()
        trace = {
            "outcome_success": False,
            "steps": [{"tool_name": "a", "error": "fail"}],
            "total_cost": 0.01,
            "total_tokens_in": 500,
            "total_tokens_out": 500,
        }
        score = scorer.score(trace)
        assert score.correctness == 0.0

    def test_rank(self):
        scorer = TrajectoryScorer()
        traces = [
            {"run_id": "t1", "outcome_success": True, "steps": [{"tool_output": "ok"}], "total_cost": 0.001, "total_tokens_in": 10, "total_tokens_out": 10},
            {"run_id": "t2", "outcome_success": False, "steps": [{"error": "fail"}], "total_cost": 0.01, "total_tokens_in": 500, "total_tokens_out": 500},
        ]
        ranked = scorer.rank(traces)
        assert len(ranked) == 2
        assert ranked[0][0]["run_id"] == "t1"


# ===========================================================================
# Elo Trajectory Rating
# ===========================================================================

class TestEloTrajectoryRating:
    def test_compare(self):
        elo = EloTrajectoryRating()
        w_rating, l_rating = elo.compare("trace_a", "trace_b")
        assert w_rating > 1500
        assert l_rating < 1500

    def test_get_rating_default(self):
        elo = EloTrajectoryRating()
        assert elo.get_rating("unknown") == 1500

    def test_leaderboard(self):
        elo = EloTrajectoryRating()
        elo.compare("a", "b")
        elo.compare("a", "c")
        lb = elo.leaderboard()
        assert lb[0][0] == "a"


# ===========================================================================
# Direct Preference Optimization
# ===========================================================================

class TestDirectPreferenceOptimizer:
    def test_optimize_updates_logits(self):
        dpo = DirectPreferenceOptimizer(beta=0.1, learning_rate=0.01)
        prefs = [("tool_a", "tool_b", 1.0)]
        logits = dpo.optimize(prefs, action_space=["tool_a", "tool_b"], n_epochs=5)
        assert "tool_a" in logits
        assert "tool_b" in logits
        assert logits["tool_a"] > logits["tool_b"]

    def test_preference_from_scores(self):
        dpo = DirectPreferenceOptimizer()
        scores = [("a", 0.9), ("b", 0.5), ("c", 0.3)]
        prefs = dpo.preference_from_scores(scores, n_pairs=5)
        assert len(prefs) > 0
        for w, l, s in prefs:
            assert isinstance(w, str)
            assert isinstance(l, str)
            assert s > 0


# ===========================================================================
# Policy
# ===========================================================================

class TestPolicy:
    def test_act_deterministic(self):
        policy = Policy(temperature=0.0, action_weights={"a": 1.0, "b": 0.0})
        assert policy.act({}, "test", ["a", "b"]) == "a"

    def test_act_fallback(self):
        policy = Policy()
        assert policy.act({}, "test") == "explore"

    def test_log_probability(self):
        policy = Policy(action_weights={"a": 0.7, "b": 0.3})
        log_p = policy.log_probability("a")
        assert log_p < 0
        log_p_b = policy.log_probability("b")
        assert log_p > log_p_b  # a is more probable → higher log prob

    def test_log_probability_of_trace(self):
        policy = Policy(action_weights={"a": 0.8, "b": 0.2})
        trace = {"steps": [{"tool_name": "a"}, {"tool_name": "a"}]}
        lp = policy.log_probability_of_trace(trace)
        assert lp < 0

    def test_update_weight(self):
        policy = Policy()
        policy.update_weight("new_action", 0.5)
        assert policy.action_weights["new_action"] == 0.5

    def test_serialization(self):
        policy = Policy(temperature=0.5, action_weights={"x": 1.0})
        d = policy.to_dict()
        restored = Policy.from_dict(d)
        assert restored.temperature == 0.5
        assert restored.action_weights["x"] == 1.0


# ===========================================================================
# Behavioral Cloning
# ===========================================================================

class TestBehavioralCloning:
    def test_clone_from_demonstrations(self):
        bc = BehavioralCloning(learning_rate=0.5, n_epochs=5)
        demos = [
            {"steps": [{"tool_name": "a"}, {"tool_name": "b"}]},
            {"steps": [{"tool_name": "a"}, {"tool_name": "c"}]},
        ]
        policy = bc.clone(demos)
        assert "a" in policy.action_weights
        assert policy.action_weights.get("a", 0) > policy.action_weights.get("b", 0)

    def test_clone_empty(self):
        bc = BehavioralCloning()
        policy = bc.clone([])
        assert isinstance(policy, Policy)


# ===========================================================================
# Value Function
# ===========================================================================

class TestValueFunction:
    def test_predict(self):
        vf = ValueFunction(state_dim=4)
        state = {"service_health": "healthy", "investigation_stage": "resolved", "steps_taken": 5, "uncertainty_level": 0.1}
        val = vf.predict(state)
        assert isinstance(val, float)

    def test_update_reduces_loss(self):
        vf = ValueFunction(state_dim=4)
        vf.learning_rate = 0.1
        states = [{"service_health": "healthy", "investigation_stage": "resolved", "steps_taken": 5, "uncertainty_level": 0.1}]
        returns = [10.0]
        loss = vf.update(states, returns)
        assert loss >= 0


# ===========================================================================
# Advantage Estimator
# ===========================================================================

class TestAdvantageEstimator:
    def test_compute_gae(self):
        ae = AdvantageEstimator()
        rewards = [1.0, 0.5, 0.0]
        values = [0.5, 0.8, 1.0]
        advantages = ae.compute_gae(rewards, values, gamma=0.99, lam=0.95)
        assert len(advantages) == 3
        assert all(isinstance(a, float) for a in advantages)

    def test_monte_carlo_returns(self):
        ae = AdvantageEstimator()
        rewards = [1.0, 1.0, 1.0]
        returns = ae.compute_monte_carlo(rewards, gamma=0.9)
        assert len(returns) == 3
        assert returns[0] > returns[1] > returns[2]


# ===========================================================================
# Exploration Scheduler
# ===========================================================================

class TestExplorationScheduler:
    def test_ucb1(self):
        exp = ExplorationScheduler()
        exp.action_values = {"a": 1.0, "b": 0.5}
        exp.action_counts = {"a": 10, "b": 2}
        exp.total_actions = 12
        score_b = exp.ucb1("b")
        score_a = exp.ucb1("a")
        assert score_b > score_a  # b has higher bonus

    def test_should_explore_high_uncertainty(self):
        exp = ExplorationScheduler()
        state = {"uncertainty_level": 0.9}
        assert exp.should_explore(state) is True

    def test_should_not_explore_low_uncertainty_with_visits(self):
        exp = ExplorationScheduler()
        state = {"uncertainty_level": 0.1, "key": "known_state"}
        exp.state_visits["known_state"] = 5
        exp.total_actions = 100
        # With many visits and low uncertainty, should exploit more often
        results = [exp.should_explore(state) for _ in range(100)]
        true_ratio = sum(results) / len(results)
        assert true_ratio < 0.5

    def test_record_action(self):
        exp = ExplorationScheduler()
        exp.record_action("tool_a", 1.0, success=True)
        assert exp.action_counts["tool_a"] == 1
        assert exp.total_actions == 1

    def test_sample_action(self):
        exp = ExplorationScheduler()
        exp.action_values = {"a": 0.5, "b": 0.8}
        exp.action_counts = {"a": 5, "b": 5}
        exp.total_actions = 10
        chosen = exp.sample_action(["a", "b"])
        assert chosen in ("a", "b")


# ===========================================================================
# Replay Buffer
# ===========================================================================

class TestReplayBuffer:
    def test_push_and_size(self):
        rb = ReplayBuffer(max_size=10)
        rb.push(Transition(state={"s": 1}, action="a", reward=1.0, next_state={"s": 2}))
        assert rb.size == 1

    def test_sample(self):
        rb = ReplayBuffer(max_size=100)
        for i in range(20):
            rb.push(Transition(state={"i": i}, action="a", reward=float(i), next_state={"i": i + 1}))
        sampled = rb.sample(5)
        assert len(sampled) == 5

    def test_max_size_eviction(self):
        rb = ReplayBuffer(max_size=5)
        for i in range(10):
            rb.push(Transition(state={"i": i}, action="a", reward=float(i), next_state={"i": i + 1}))
        assert rb.size == 5

    def test_clear(self):
        rb = ReplayBuffer(max_size=10)
        rb.push(Transition(state={}, action="a", reward=0.0, next_state={}))
        rb.clear()
        assert rb.size == 0

    def test_latest(self):
        rb = ReplayBuffer(max_size=10)
        for i in range(5):
            rb.push(Transition(state={"i": i}, action="a", reward=float(i), next_state={"i": i + 1}))
        latest = rb.latest(3)
        assert len(latest) == 3
        assert latest[-1].state["i"] == 4


# ===========================================================================
# RL-Enabled Agent
# ===========================================================================

class TestRLEnabledAgent:
    def test_select_action(self):
        agent = RLEnabledAgent(domain="test")
        agent.policy = Policy(action_weights={"investigate": 1.0, "fix": 0.5})
        action = agent.select_action({"uncertainty_level": 0.1, "key": "known"})
        assert action in ("investigate", "fix", "explore")

    def test_on_step_complete_adds_to_buffer(self):
        agent = RLEnabledAgent(domain="test")
        before = {"uncertainty_level": 0.5, "tools_used": []}
        after = {"uncertainty_level": 0.3, "tools_used": ["s"]}
        step = StepRecord(step_number=1, tool_name="search", tokens_in=50, tokens_out=50)
        agent.on_step_start(before)
        agent.on_step_complete(step, before, after)
        assert agent.replay_buffer.size >= 1

    def test_finish_trajectory(self):
        agent = RLEnabledAgent(domain="test")
        trace = {
            "run_id": "test_run",
            "outcome_success": True,
            "steps": [{"tool_name": "a", "tool_output": "ok"}],
            "total_cost": 0.001,
            "total_tokens_in": 10,
            "total_tokens_out": 10,
        }
        agent.finish_trajectory(trace)
        assert agent.elo.get_rating("test_run") > 0

    def test_end_to_end_cycle(self):
        agent = RLEnabledAgent(domain="test")
        state = {"investigation_stage": "not_started", "service_health": "unknown", "uncertainty_level": 0.5, "tools_used": []}
        agent.on_step_start(state)
        step = StepRecord(step_number=1, tool_name="investigate", tokens_in=100, tokens_out=50)
        next_state = {"investigation_stage": "gathering_data", "service_health": "degraded", "uncertainty_level": 0.4, "tools_used": ["investigate"]}
        agent.on_step_complete(step, state, next_state)

        agent.on_step_start(next_state)
        step2 = StepRecord(step_number=2, tool_name="fix", tokens_in=100, tokens_out=50, error="")
        final_state = {"investigation_stage": "resolved", "service_health": "healthy", "uncertainty_level": 0.1, "tools_used": ["investigate", "fix"], "is_terminal_success": True}
        agent.on_step_complete(step2, next_state, final_state)

        trace = {
            "run_id": "e2e",
            "outcome_success": True,
            "steps": [{"tool_name": "investigate"}, {"tool_name": "fix"}],
            "total_cost": 0.002,
            "total_tokens_in": 200,
            "total_tokens_out": 100,
        }
        agent.finish_trajectory(trace)
        assert agent.replay_buffer.size == 2


# ===========================================================================
# Trajectory Score Serialization
# ===========================================================================

class TestTrajectoryScore:
    def test_composite_weighted(self):
        ts = TrajectoryScore(steps_efficiency=1.0, correctness=1.0)
        assert ts.composite > 0.3
        assert ts.composite < 1.0
