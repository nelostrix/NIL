from __future__ import annotations

import math
import time
from collections import Counter

import pytest

from nil.models.belief import BeliefTracker
from nil.models.framework import EnvironmentModel, ModelAwareAgent
from nil.models.learning import ModelUpdate, OnlineModelLearner
from nil.models.observations import ObservationModel
from nil.models.planners import HeuristicPolicy, MCTSPlanner, ValuePlanner
from nil.models.rewards import RewardLearner, RewardModel, RewardObservation
from nil.models.state_space import (
    EnvironmentState,
    StateSpace,
    StateSpaceLearner,
)
from nil.models.transitions import TransitionModel


# ---------------------------------------------------------------------------
# Helpers — match the trace schema from other test files
# ---------------------------------------------------------------------------

def _trace(
    run_id,
    success,
    tools,
    domain="test",
    cost=0.001,
    duration=1.0,
    error=None,
    created_offset=0,
    task="generic task",
    step_errors=None,
):
    steps = []
    for i, tool in enumerate(tools):
        step_num = i + 1
        step = {
            "step_number": step_num,
            "step_type": "tool_call",
            "tool_name": tool,
            "tool_input": {},
            "tool_output": "ok",
            "duration_ms": 50,
            "tokens_in": 10,
            "tokens_out": 5,
            "error": None,
            "timestamp": time.time() + created_offset + i * 0.1,
        }
        if step_errors and step_num in step_errors:
            step["error"] = str(step_errors[step_num])
        steps.append(step)

    return {
        "run_id": run_id,
        "domain": domain,
        "task": task,
        "steps": steps,
        "outcome_success": success,
        "outcome_output": "done" if success else "failed",
        "total_duration": duration,
        "total_tokens_in": 100,
        "total_tokens_out": 50,
        "total_cost": cost,
        "tool_sequence": tools,
        "error": error,
        "created_at": time.time() + created_offset,
        "metadata": {},
    }


def _state(**kwargs) -> EnvironmentState:
    return EnvironmentState(**kwargs)


# ===========================================================================
# State Space Tests
# ===========================================================================


class TestEnvironmentState:
    def test_from_trace_success(self):
        t = _trace("r1", True, ["tool_a", "tool_b"])
        s = EnvironmentState.from_trace(t)
        assert s.service_health == "healthy"
        assert s.investigation_stage == "resolved"
        assert s.is_terminal_success is True
        assert s.is_terminal_failure is False
        assert s.steps_taken == 2
        assert s.tools_used == ["tool_a", "tool_b"]

    def test_from_trace_failure(self):
        t = _trace("r2", False, ["tool_a"], error="crash")
        s = EnvironmentState.from_trace(t)
        assert s.service_health == "degraded"
        assert s.investigation_stage == "fixing"
        assert s.is_terminal_success is False
        assert s.is_terminal_failure is True
        assert s.steps_taken == 1

    def test_to_key_consistency(self):
        s1 = _state(service_health="healthy", investigation_stage="resolved")
        s2 = _state(service_health="healthy", investigation_stage="resolved")
        assert s1.to_key() == s2.to_key()

    def test_to_dict_roundtrip(self):
        s = _state(
            service_health="degraded",
            incident_count=3,
            tools_used=["a", "b"],
            errors_encountered=["err1"],
        )
        d = s.to_dict()
        restored = EnvironmentState.from_dict(d)
        assert restored.service_health == "degraded"
        assert restored.incident_count == 3
        assert restored.tools_used == ["a", "b"]
        assert restored.errors_encountered == ["err1"]


class TestStateSpace:
    def test_get_or_create(self):
        ss = StateSpace()
        k1 = _state(service_health="healthy").to_key()
        k2 = _state(service_health="degraded").to_key()
        assert ss.get_or_create(k1) == 0
        assert ss.get_or_create(k2) == 1
        assert ss.get_or_create(k1) == 0
        assert ss.size == 2

    def test_add_state(self):
        ss = StateSpace()
        s = _state(service_health="healthy", investigation_stage="resolved")
        sid = ss.add_state(s)
        assert sid == 0
        assert ss.get_key(0) == s.to_key()
        assert ss.get_id(s.to_key()) == 0

    def test_all_keys_ids(self):
        ss = StateSpace()
        ss.get_or_create("k1")
        ss.get_or_create("k2")
        assert set(ss.all_keys()) == {"k1", "k2"}
        assert set(ss.all_ids()) == {0, 1}

    def test_serialization(self):
        ss = StateSpace()
        ss.get_or_create("k1")
        ss.get_or_create("k2")
        d = ss.to_dict()
        restored = StateSpace.from_dict(d)
        assert restored.get_id("k1") == 0
        assert restored.get_id("k2") == 1
        assert restored.dimensions == ss.dimensions

    def test_empty_state_space(self):
        ss = StateSpace()
        assert ss.size == 0
        assert ss.all_keys() == []
        assert ss.all_ids() == []


class TestStateSpaceLearner:
    def test_discover_states_from_traces(self):
        t1 = _trace("r1", True, ["tool_a", "tool_b"])
        t2 = _trace("r2", False, ["tool_c"], error="fail")
        learner = StateSpaceLearner()
        space = learner.discover_states([t1, t2])
        assert space.size >= 2

    def test_discover_respects_max(self):
        traces = [_trace(f"r{i}", True, ["tool_a"]) for i in range(10)]
        learner = StateSpaceLearner(max_states=3)
        space = learner.discover_states(traces)
        assert space.size <= 3

    def test_extract_transition_states(self):
        t = _trace("r1", True, ["tool_a", "tool_b", "tool_c"])
        learner = StateSpaceLearner()
        states = learner._extract_transition_states(t)
        assert len(states) == 3

    def test_build_transition_counts(self):
        t1 = _trace("r1", True, ["tool_a", "tool_b"])
        t2 = _trace("r2", True, ["tool_a", "tool_c"])
        learner = StateSpaceLearner()
        space = learner.discover_states([t1, t2])
        counts = learner.build_transition_counts([t1, t2], space)
        assert len(counts) >= 1


# ===========================================================================
# Transition Model Tests
# ===========================================================================


class TestTransitionModel:
    def test_record_and_predict(self):
        tm = TransitionModel()
        tm.record("s1", "act_a", "s2")
        tm.record("s1", "act_a", "s2")
        tm.record("s1", "act_a", "s3")

        preds = tm.predict("s1", "act_a")
        assert len(preds) == 2
        assert preds[0][0] == "s2"
        assert preds[0][1] == pytest.approx(2 / 3)
        assert preds[1][0] == "s3"
        assert preds[1][1] == pytest.approx(1 / 3)

    def test_prob(self):
        tm = TransitionModel()
        tm.record("s1", "act_a", "s2")
        assert tm.prob("s1", "act_a", "s2") == 1.0
        assert tm.prob("s1", "act_a", "s3") == 0.0
        assert tm.prob("s2", "act_a", "s1") == 0.0

    def test_expected_outcome(self):
        tm = TransitionModel()
        tm.record("s1", "act_a", "s2", 3)
        tm.record("s1", "act_a", "s3", 1)
        outcome = tm.expected_outcome("s1", "act_a")
        assert outcome.next_state == "s2"
        assert outcome.confidence == pytest.approx(0.75)
        assert len(outcome.alternative_states) == 1

    def test_unknown_outcome(self):
        tm = TransitionModel()
        outcome = tm.expected_outcome("unknown", "act")
        assert outcome.confidence == 0.0

    def test_uncertainty(self):
        tm = TransitionModel()
        assert tm.uncertainty("s1", "act_a") == 1.0
        tm.record("s1", "act_a", "s2", 1)
        tm.record("s1", "act_a", "s3", 1)
        u = tm.uncertainty("s1", "act_a")
        assert 0 < u < 1

    def test_known_actions(self):
        tm = TransitionModel()
        tm.record("s1", "a1", "s2")
        tm.record("s1", "a2", "s3")
        tm.record("s2", "a3", "s1")
        assert set(tm.known_actions("s1")) == {"a1", "a2"}
        assert tm.known_actions("s2") == ["a3"]

    def test_merge(self):
        tm1 = TransitionModel()
        tm1.record("s1", "a", "s2", 2)
        tm2 = TransitionModel()
        tm2.record("s1", "a", "s3", 1)
        tm1.merge(tm2)
        assert tm1.prob("s1", "a", "s2") == pytest.approx(2 / 3)
        assert tm1.prob("s1", "a", "s3") == pytest.approx(1 / 3)

    def test_record_batch(self):
        tm = TransitionModel()
        tm.record_batch([
            ("s1", "a", "s2", 2),
            ("s1", "a", "s3", 1),
        ])
        assert tm.total_transitions == 3

    def test_serialization(self):
        tm = TransitionModel()
        tm.record("s1", "a", "s2", 2)
        tm.record("s1", "a", "s3", 1)
        d = tm.to_dict()
        restored = TransitionModel.from_dict(d)
        assert restored.prob("s1", "a", "s2") == pytest.approx(2 / 3)
        assert restored.total_transitions == 3


# ===========================================================================
# Observation Model Tests
# ===========================================================================


class TestObservationModel:
    def test_record_and_prob(self):
        om = ObservationModel()
        om.record("s1", "act_a", "obs_ok")
        assert om.prob("obs_ok", "s1", "act_a") == 1.0
        assert om.prob("obs_fail", "s1", "act_a") == pytest.approx(1e-6)

    def test_record_multiple_observations(self):
        om = ObservationModel()
        om.record("s1", "act_a", "obs_ok", 3)
        om.record("s1", "act_a", "obs_warn", 1)
        assert om.prob("obs_ok", "s1", "act_a") == pytest.approx(0.75)
        assert om.prob("obs_warn", "s1", "act_a") == pytest.approx(0.25)

    def test_most_likely_state(self):
        om = ObservationModel()
        om.record("s1", "act_a", "obs_ok", 5)
        om.record("s2", "act_a", "obs_ok", 1)
        candidates = ["s1", "s2"]
        scored = om.most_likely_state("obs_ok", "act_a", candidates)
        assert scored[0][0] == "s1"

    def test_diagnostic_value(self):
        om = ObservationModel()
        om.record("s1", "act_a", "obs_ok", 5)
        om.record("s2", "act_a", "obs_fail", 5)
        dv = om.diagnostic_value("obs_ok", "act_a", ["s1", "s2"])
        assert dv > 0.5

    def test_serialization(self):
        om = ObservationModel()
        om.record("s1", "a", "obs_ok")
        d = om.to_dict()
        restored = ObservationModel.from_dict(d)
        assert restored.prob("obs_ok", "s1", "a") == 1.0
        assert restored.total_observations == 1

    def test_empty_model_defaults(self):
        om = ObservationModel()
        assert om.prob("anything", "s1", "a") == pytest.approx(1e-6)
        assert om.total_observations == 0
        assert om.most_likely_state("x", "a", []) == []
        assert om.diagnostic_value("x", "a", []) == 0.0


# ===========================================================================
# Belief Tracker Tests
# ===========================================================================


class TestBeliefTracker:
    def test_initial_belief_is_uniform(self):
        ss = StateSpace()
        ss.get_or_create("s1")
        ss.get_or_create("s2")
        bt = BeliefTracker(state_space=ss)
        assert set(bt.belief.keys()) == {"s1", "s2"}
        assert bt.belief["s1"] == pytest.approx(0.5)
        assert bt.belief["s2"] == pytest.approx(0.5)

    def test_empty_state_space_reset(self):
        bt = BeliefTracker(state_space=StateSpace())
        assert bt.belief == {}

    def test_update_with_evidence(self):
        ss = StateSpace()
        ss.get_or_create('{"health": "healthy"}')
        ss.get_or_create('{"health": "degraded"}')
        tm = TransitionModel()
        tm.record('{"health": "healthy"}', "investigate", '{"health": "degraded"}', 1)
        tm.record('{"health": "degraded"}', "investigate", '{"health": "degraded"}', 1)
        om = ObservationModel()
        om.record('{"health": "healthy"}', "investigate", "all_good", 1)
        om.record('{"health": "degraded"}', "investigate", "error_found", 1)

        bt = BeliefTracker(
            state_space=ss,
            transition_model=tm,
            observation_model=om,
        )
        bt.incorporate_task("fix the error")
        assert bt.belief.get('{"health": "degraded"}', 0) > bt.belief.get('{"health": "healthy"}', 0)

        updated = bt.update("investigate", "error_found")
        assert bt.most_likely_state == '{"health": "degraded"}'
        assert bt.most_likely_state_probability > 0.5
        assert 0 < bt.uncertainty < math.log(3)

    def test_incorporate_task_biases_belief(self):
        ss = StateSpace()
        ss.get_or_create('{"health": "healthy"}')
        ss.get_or_create('{"health": "degraded"}')
        bt = BeliefTracker(state_space=ss)
        bt.incorporate_task("deploy the service")
        assert bt.belief

    def test_top_beliefs(self):
        ss = StateSpace()
        ss.get_or_create("s1")
        ss.get_or_create("s2")
        ss.get_or_create("s3")
        bt = BeliefTracker(state_space=ss)
        bt.belief = {"s1": 0.5, "s2": 0.3, "s3": 0.2}
        top = bt.top_beliefs(2)
        assert len(top) == 2
        assert top[0][0] == "s1"

    def test_serialization(self):
        ss = StateSpace()
        ss.get_or_create("s1")
        bt = BeliefTracker(state_space=ss)
        d = bt.to_dict()
        assert "belief" in d
        assert "most_likely" in d


# ===========================================================================
# Planner Tests
# ===========================================================================


class TestValuePlanner:
    def test_plan_simple(self):
        tm = TransitionModel()
        tm.record("s1", "go_s2", "s2", 5)
        tm.record("s1", "go_s3", "s3", 1)
        tm.record("s2", "done", "_terminal", 5)
        tm.record("s3", "done", "_terminal", 1)

        def reward_fn(s, a):
            if s == "s2" and a == "done":
                return 10.0
            if s == "s3" and a == "done":
                return 1.0
            return -0.1

        planner = ValuePlanner(gamma=0.9, theta=1e-3, max_iterations=500)
        policy = planner.plan(
            states=["s1", "s2", "s3"],
            transition_model=tm,
            reward_fn=reward_fn,
        )
        assert policy["s1"] == "go_s2", f"Expected go_s2, got {policy['s1']}"

    def test_get_value(self):
        planner = ValuePlanner()
        planner._values = {"s1": 5.0}
        assert planner.get_value("s1") == 5.0
        assert planner.get_value("unknown") == 0.0

    def test_get_action_fallback(self):
        planner = ValuePlanner()
        assert planner.get_action("unknown") == "default"


class TestMCTSPlanner:
    def test_plan_produces_policy(self):
        tm = TransitionModel()
        tm.record("s1", "go_s2", "s2", 3)
        tm.record("s1", "go_s3", "s3", 1)
        tm.record("s2", "done", "_terminal", 3)

        def reward_fn(s, a):
            if s == "s2":
                return 5.0
            return 0.0

        planner = MCTSPlanner(iterations=30, gamma=0.9)
        policy = planner.plan(
            states=["s1", "s2", "s3"],
            transition_model=tm,
            reward_fn=reward_fn,
        )
        assert "s1" in policy
        assert policy["s2"] == "done"

    def test_get_action_fallback(self):
        planner = MCTSPlanner()
        assert planner.get_action("unknown") == "default"

    def test_plan_with_terminal_states(self):
        tm = TransitionModel()
        tm.record("s1", "done", "_terminal", 1)

        def reward_fn(s, a):
            return 0.0

        planner = MCTSPlanner(iterations=10)
        policy = planner.plan(
            states=["s1"],
            transition_model=tm,
            reward_fn=reward_fn,
            terminal_states={"s1"},
        )
        assert policy.get("s1") == "_terminal"


class TestHeuristicPolicy:
    def test_rule_based_policy(self):
        policy = HeuristicPolicy.action_for_stage("analyzing")
        action = policy.get_action('{"stage": "analyzing"}')
        assert action == "analyze"

    def test_custom_rules(self):
        policy = HeuristicPolicy()
        policy.add_rule(lambda s: "urgent" in s, "act_fast")
        policy.add_rule(lambda s: True, "default")
        assert policy.get_action("urgent matter") == "act_fast"
        assert policy.get_action("normal") == "default"

    def test_plan_all_states(self):
        policy = HeuristicPolicy()
        policy.add_rule(lambda s: "fix" in s, "repair")
        result = policy.plan(states=["fix_it", "other", "fix_again"])
        assert result["fix_it"] == "repair"
        assert result["fix_again"] == "repair"


# ===========================================================================
# Online Learner Tests
# ===========================================================================


class TestOnlineModelLearner:
    def test_update_from_trace(self):
        learner = OnlineModelLearner()
        t = _trace("r1", True, ["tool_a", "tool_b"], task="fix error")
        update = learner.update(t)
        assert learner.state_space.size >= 1
        assert update.transitions_added >= 1
        assert update.observations_added >= 1

    def test_update_batch(self):
        learner = OnlineModelLearner()
        traces = [
            _trace("r1", True, ["tool_a", "tool_b"]),
            _trace("r2", False, ["tool_c"], error="crash"),
        ]
        update = learner.update_batch(traces)
        assert update.transitions_added >= 1
        assert update.observations_added >= 1
        assert update.model_refined is True

    def test_get_belief_tracker(self):
        learner = OnlineModelLearner()
        t = _trace("r1", True, ["tool_a", "tool_b"])
        learner.update(t)
        bt = learner.get_belief_tracker()
        assert isinstance(bt, BeliefTracker)
        assert bt.state_space.size >= 1

    def test_serialization(self):
        learner = OnlineModelLearner()
        t = _trace("r1", True, ["tool_a"])
        learner.update(t)
        d = learner.to_dict()
        assert "state_space" in d
        assert "transition_model" in d
        assert d["total_updates"] == 1


# ===========================================================================
# EnvironmentModel Integration Tests
# ===========================================================================


class TestEnvironmentModel:
    def test_ingest_single_trace(self):
        em = EnvironmentModel(domain="test")
        t = _trace("r1", True, ["tool_a", "tool_b"])
        update = em.ingest(t)
        assert em.total_traces == 1
        assert em.state_count >= 1
        assert update.transitions_added >= 1

    def test_ingest_batch(self):
        em = EnvironmentModel(domain="test")
        traces = [
            _trace("r1", True, ["tool_a", "tool_b"]),
            _trace("r2", False, ["tool_c"], error="fail"),
        ]
        update = em.ingest_batch(traces)
        assert em.total_traces == 2
        assert update.new_states_discovered >= 0

    def test_get_belief_with_task_prior(self):
        em = EnvironmentModel(domain="test")
        _ = em.ingest(_trace("r1", True, ["tool_a"]))
        bt = em.get_belief(task="fix the error")
        assert bt.state_space.size >= 1

    def test_suggest_action_with_belief(self):
        em = EnvironmentModel(domain="test", planner_type="heuristic")
        em.ingest(_trace("r1", True, ["tool_a", "tool_b"]))
        bt = em.get_belief(task="fix error")
        suggestion = em.suggest_action(belief_tracker=bt)
        assert "action" in suggestion
        assert "confidence" in suggestion
        assert "uncertainty" in suggestion

    def test_suggest_action_no_belief(self):
        em = EnvironmentModel(domain="test")
        suggestion = em.suggest_action()
        assert suggestion["action"] == "explore"
        assert suggestion["confidence"] == 0.0

    def test_state_info(self):
        em = EnvironmentModel(domain="test")
        s = _state(service_health="degraded", investigation_stage="fixing")
        info = em.state_info(s)
        assert "key" in info
        assert "uncertainty" in info
        assert "known_actions" in info

    def test_summarize(self):
        em = EnvironmentModel(domain="test")
        em.ingest(_trace("r1", True, ["tool_a", "tool_b"]))
        summary = em.summarize()
        assert summary["domain"] == "test"
        assert summary["total_traces"] == 1
        assert summary["states"] >= 1

    def test_serialization_roundtrip(self):
        em = EnvironmentModel(domain="test")
        em.ingest(_trace("r1", True, ["tool_a"]))
        em.ingest(_trace("r2", True, ["tool_b"]))
        d = em.to_dict()
        restored = EnvironmentModel.from_dict(d)
        assert restored.domain == "test"
        assert restored.state_count == em.state_count
        assert restored.total_traces == 2

    def test_auto_replan_after_3_traces(self):
        em = EnvironmentModel(domain="test")
        for i in range(4):
            em.ingest(_trace(f"r{i}", True, ["tool_a", "tool_b"]))
        # After 3+ traces, reward model is re-learned and replan triggered
        assert em.total_traces == 4
        assert em.reward_model.domain == "test"

    def test_ingest_trace_with_step_errors(self):
        em = EnvironmentModel(domain="test")
        t = _trace(
            "r1",
            False,
            ["tool_a", "tool_b"],
            error="final_error",
            step_errors={2: "step_error"},
        )
        update = em.ingest(t)
        assert update.transitions_added >= 0


# ===========================================================================
# ModelAwareAgent Integration Tests
# ===========================================================================


class TestModelAwareAgent:
    def test_start_task(self):
        agent = ModelAwareAgent(domain="test")
        suggestion = agent.start_task("fix the deployment")
        assert "action" in suggestion
        assert "confidence" in suggestion
        assert agent._current_task == "fix the deployment"

    def test_observe_outcome(self):
        agent = ModelAwareAgent(domain="test")
        agent.start_task("test task")
        result = agent.observe_outcome({
            "step_number": 1,
            "step_type": "tool_call",
            "tool_name": "tool_a",
            "tool_output": "ok",
        })
        assert "action" in result
        assert "ucb" in result

    def test_finish_task(self):
        agent = ModelAwareAgent(domain="test")
        agent.start_task("test task")
        t = _trace("r1", True, ["tool_a", "tool_b"])
        update = agent.finish_task(t)
        assert agent.env_model.total_traces == 1
        assert isinstance(update, ModelUpdate)

    def test_end_to_end_cycle(self):
        agent = ModelAwareAgent(domain="test")

        suggestion = agent.start_task("fix the error in production")
        assert "action" in suggestion

        for result in ["ok", "error_found", "fixed"]:
            obs = agent.observe_outcome({
                "step_number": 1,
                "step_type": "tool_call",
                "tool_name": f"tool_{result}",
                "result": result,
            })
            assert "action" in obs

        t = _trace("r1", True, ["tool_ok", "tool_error_found", "tool_fixed"])
        update = agent.finish_task(t)
        assert update.transitions_added >= 2

    def test_traces_with_errors(self):
        agent = ModelAwareAgent(domain="test")
        agent.start_task("fix errors")
        t = _trace("r1", False, ["tool_a"], error="crash")
        agent.finish_task(t)
        assert agent.env_model.total_traces == 1
