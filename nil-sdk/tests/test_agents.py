from __future__ import annotations

import time
import uuid

import pytest

from nil.agents.blackboard import Blackboard
from nil.agents.consensus import (
    AgentOpinion,
    AgentProposal,
    ConsensusMechanism,
    ConsensusResult,
)
from nil.agents.framework import MultiAgentNil
from nil.agents.market import MarketCoordinator, Subtask
from nil.agents.messages import AgentMessage, MessageBus, MessageType
from nil.agents.orchestrator import OrchestratorAgent, Phase, Subtask as OrchSubtask
from nil.agents.pool import AgentPool
from nil.agents.roles import AgentResult, AgentRole, AgentSpec, MultiAgentOutcome
from nil.agents.teams import AgentTeam


# ===========================================================================
# Roles / AgentSpec
# ===========================================================================

class TestAgentSpec:
    def test_default_prompt_by_role(self):
        spec = AgentSpec(role=AgentRole.INVESTIGATOR)
        assert "investigator" in spec.system_prompt.lower()
        assert "search" in spec.system_prompt.lower()

    def test_default_tools_by_role(self):
        spec = AgentSpec(role=AgentRole.EXECUTOR)
        assert "restart_service" in spec.tools

    def test_default_name_from_role(self):
        spec = AgentSpec(role=AgentRole.ORCHESTRATOR)
        assert spec.name == "orchestrator"

    def test_custom_values(self):
        spec = AgentSpec(
            role=AgentRole.ANALYST,
            name="custom-analyst",
            model="claude-opus-4-20250514",
            temperature=0.3,
            max_tokens=4096,
        )
        assert spec.name == "custom-analyst"
        assert spec.model == "claude-opus-4-20250514"
        assert spec.temperature == 0.3
        assert spec.max_tokens == 4096

    def test_all_roles_have_prompts(self):
        for role in AgentRole:
            spec = AgentSpec(role=role)
            assert spec.system_prompt, f"Missing prompt for {role}"


# ===========================================================================
# Blackboard
# ===========================================================================

class TestBlackboard:
    def test_write_and_read(self):
        bb = Blackboard()
        bb.write("test.key", {"data": 42})
        assert bb.read("test.key") == {"data": 42}

    def test_read_default(self):
        bb = Blackboard()
        assert bb.read("nonexistent", "default") == "default"

    def test_append(self):
        bb = Blackboard()
        bb.append("list.key", "a")
        bb.append("list.key", "b")
        assert bb.read("list.key") == ["a", "b"]

    def test_append_initializes_list(self):
        bb = Blackboard()
        bb.append("new.key", "first")
        assert bb.read("new.key") == ["first"]

    def test_namespace(self):
        bb = Blackboard()
        bb.write("incident.1.status", "open")
        bb.write("incident.2.status", "closed")
        bb.write("other.key", "value")
        ns = bb.namespace("incident.")
        assert len(ns) == 2

    def test_search_by_key(self):
        bb = Blackboard()
        bb.write("incident-4421", "high priority")
        results = bb.search("4421")
        assert len(results) == 1

    def test_search_by_value(self):
        bb = Blackboard()
        bb.write("note", "urgent resolution needed")
        results = bb.search("urgent")
        assert len(results) == 1

    def test_clear(self):
        bb = Blackboard()
        bb.write("k1", "v1")
        bb.write("k2", "v2")
        bb.clear()
        assert len(bb.store) == 0

    def test_subscription_notification(self):
        bb = Blackboard()
        notified = []

        def callback(entry):
            notified.append(entry.key)

        bb.subscribe("watch_key", callback)
        bb.write("watch_key", "value")
        assert "watch_key" in notified

    def test_unsubscribe(self):
        bb = Blackboard()
        notified = []

        def callback(entry):
            notified.append(entry.key)

        sub_id = bb.subscribe("key", callback)
        bb.write("key", "v1")
        bb.unsubscribe(sub_id)
        bb.write("key", "v2")
        assert len(notified) == 1

    def test_read_entry(self):
        bb = Blackboard()
        bb.write("k", "v", writer="alice")
        entry = bb.read_entry("k")
        assert entry is not None
        assert entry.writer == "alice"
        assert entry.value == "v"

    def test_to_dict(self):
        bb = Blackboard()
        bb.write("a", 1)
        d = bb.to_dict()
        assert "a" in d


# ===========================================================================
# Messages / MessageBus
# ===========================================================================

class TestMessageBus:
    def test_send_and_receive(self):
        bus = MessageBus(agent_name="test")
        bus.queues["alice"] = []
        msg = AgentMessage(source="test", target="alice", content="hello")
        bus.send(msg)
        received = bus.receive("alice")
        assert len(received) == 1
        assert received[0].content == "hello"

    def test_broadcast(self):
        bus = MessageBus(agent_name="sender")
        bus.queues["a"] = []
        bus.queues["b"] = []
        msg = AgentMessage(source="sender", target="broadcast", content="hi")
        bus.send(msg)
        assert len(bus.receive("a")) == 1
        assert len(bus.receive("b")) == 1

    def test_request_response(self):
        bus = MessageBus(agent_name="requester")
        bus.queues["responder"] = []

        def handle():
            msgs = bus.receive("responder")
            if msgs:
                bus.respond(msgs[0], "response_data")

        result = bus.request("responder", "query")
        handle()
        assert result["received"] is True

    def test_subscribe_publish(self):
        bus = MessageBus()
        received = []

        def cb(data):
            received.append(data)

        bus.subscribe("topic_a", cb)
        bus.publish("topic_a", "data")
        assert "data" in received

    def test_history(self):
        bus = MessageBus()
        bus.queues["bob"] = []
        bus.send(AgentMessage(source="alice", target="bob", content="msg1"))
        bus.send(AgentMessage(source="alice", target="bob", content="msg2"))
        assert len(bus._history) == 2


# ===========================================================================
# AgentPool
# ===========================================================================

class TestAgentPool:
    def test_register_and_spawn(self):
        pool = AgentPool()
        spec = AgentSpec(role=AgentRole.INVESTIGATOR, name="inv-1")
        pool.register_spec(spec)
        agent = pool.spawn(AgentRole.INVESTIGATOR)
        assert agent.role == AgentRole.INVESTIGATOR
        assert agent.spec.name == "inv-1"

    def test_acquire_release(self):
        pool = AgentPool()
        pool.register_spec(AgentSpec(role=AgentRole.EXECUTOR, name="exec-1"))
        agent = pool.acquire(AgentRole.EXECUTOR)
        assert agent is not None
        pool.release(agent)
        same = pool.acquire(AgentRole.EXECUTOR)
        assert same is agent  # pool reuses

    def test_spawn_unknown_role(self):
        pool = AgentPool()
        with pytest.raises(ValueError):
            pool.spawn(AgentRole.ANALYST)

    def test_list_agents(self):
        pool = AgentPool()
        pool.register_spec(AgentSpec(role=AgentRole.MONITOR))
        pool.register_spec(AgentSpec(role=AgentRole.CRITIC))
        agents = pool.list_agents()
        roles = {a["role"] for a in agents}
        assert AgentRole.MONITOR.value in roles
        assert AgentRole.CRITIC.value in roles

    def test_agent_run(self):
        pool = AgentPool()
        pool.register_spec(AgentSpec(role=AgentRole.ANALYST))
        agent = pool.spawn(AgentRole.ANALYST)
        result = agent.run(OrchSubtask(description="analyze data"))
        assert result.success
        assert "analyze data" in result.output


# ===========================================================================
# Orchestrator
# ===========================================================================

class TestOrchestratorAgent:
    def test_run_returns_outcome(self):
        pool = AgentPool()
        pool.register_spec(AgentSpec(role=AgentRole.INVESTIGATOR, name="inv"))
        pool.register_spec(AgentSpec(role=AgentRole.ANALYST, name="anl"))
        pool.register_spec(AgentSpec(role=AgentRole.EXECUTOR, name="exe"))
        pool.register_spec(AgentSpec(role=AgentRole.VERIFIER, name="ver"))
        orch = OrchestratorAgent(domain="test", agent_pool=pool)
        outcome = orch.run("Resolve API latency spike on /v2/users")
        assert isinstance(outcome, MultiAgentOutcome)
        assert outcome.task == "Resolve API latency spike on /v2/users"
        assert outcome.total_duration > 0

    def test_decompose_creates_plan(self):
        orch = OrchestratorAgent()
        plan = orch._decompose("test task")
        assert len(plan.phases) >= 3

    def test_verify_phase_all_success(self):
        orch = OrchestratorAgent()
        phase = Phase(id="test")
        results = {"a": AgentResult(subtask_id="a", success=True)}
        assert orch._verify_phase(phase, results) is True

    def test_verify_phase_has_failure(self):
        orch = OrchestratorAgent()
        phase = Phase(id="test")
        results = {"a": AgentResult(subtask_id="a", success=False)}
        assert orch._verify_phase(phase, results) is False

    def test_recover_phase(self):
        pool = AgentPool()
        pool.register_spec(AgentSpec(role=AgentRole.ANALYST))
        orch = OrchestratorAgent(domain="test", agent_pool=pool)
        phase = Phase(subtasks=[
            OrchSubtask(id="fail1", description="retry me", assigned_role=AgentRole.ANALYST),
        ])
        results = {"fail1": AgentResult(subtask_id="fail1", success=False)}
        recovered = orch._recover_phase(phase, results)
        assert len(recovered) >= 1

    def test_synthesize(self):
        orch = OrchestratorAgent()
        results = {
            "a": AgentResult(subtask_id="a", success=True),
            "b": AgentResult(subtask_id="b", success=True),
        }
        summary = orch._synthesize(results)
        assert "2/2" in summary

    def test_multi_agent_outcome(self):
        outcome = MultiAgentOutcome(
            task="test",
            final_output="done",
            total_cost=0.05,
            total_duration=1.0,
            success=True,
        )
        assert outcome.success
        assert outcome.total_cost == 0.05


# ===========================================================================
# Consensus
# ===========================================================================

class TestConsensusMechanism:
    def test_majority_vote(self):
        cm = ConsensusMechanism()
        proposals = [
            AgentProposal(id="a", label="Option A"),
            AgentProposal(id="b", label="Option B"),
        ]
        voters = [
            AgentSpec(role=AgentRole.ANALYST, name="alice"),
            AgentSpec(role=AgentRole.ANALYST, name="bob"),
            AgentSpec(role=AgentRole.ANALYST, name="charlie"),
        ]
        evaluations = {"alice": "a", "bob": "a", "charlie": "b"}
        result = cm.majority_vote(proposals, voters, evaluations)
        assert result.winner == "a"
        assert result.vote_share > 0.5

    def test_majority_vote_tie(self):
        cm = ConsensusMechanism()
        proposals = [AgentProposal(id="x"), AgentProposal(id="y")]
        voters = [AgentSpec(role=AgentRole.ANALYST, name="a")]
        evaluations = {"a": "x"}
        result = cm.majority_vote(proposals, voters, evaluations)
        assert result.winner == "x"

    def test_debate(self):
        cm = ConsensusMechanism()
        outcome = cm.debate("Deploy now", "Roll back", rounds=2)
        assert outcome.winner in ("proponent", "opponent")
        assert outcome.confidence >= 0
        assert len(outcome.transcript) == 2

    def test_confidence_weighted_consensus(self):
        cm = ConsensusMechanism()
        opinions = [
            AgentOpinion(value=0.8, confidence=0.9, agent_name="expert_a"),
            AgentOpinion(value=0.6, confidence=0.5, agent_name="expert_b"),
        ]
        result = cm.confidence_weighted_consensus(opinions)
        assert result.n_agents == 2
        assert result.confidence > 0

    def test_confidence_weighted_empty(self):
        cm = ConsensusMechanism()
        result = cm.confidence_weighted_consensus([])
        assert result.n_agents == 0

    def test_is_conclusive(self):
        r = ConsensusResult(vote_share=0.6)
        assert r.is_conclusive() is True
        r2 = ConsensusResult(vote_share=0.3)
        assert r2.is_conclusive() is False


# ===========================================================================
# Market Coordinator
# ===========================================================================

class TestMarketCoordinator:
    def test_assign_returns_winner(self):
        from nil.agents.pool import AgentPool
        pool = AgentPool()
        pool.register_spec(AgentSpec(role=AgentRole.INVESTIGATOR, name="inv-1"))
        pool.register_spec(AgentSpec(role=AgentRole.EXECUTOR, name="exec-1"))
        agents = [
            pool.spawn(AgentRole.INVESTIGATOR),
            pool.spawn(AgentRole.EXECUTOR),
        ]
        mc = MarketCoordinator()
        subtask = Subtask(description="investigate logs", required_capabilities=["search_logs"])
        winner = mc.assign(subtask, agents)
        assert winner is not None
        assert winner.name in ("inv-1-1", "exec-1-1")

    def test_capability_match(self):
        from nil.agents.pool import AgentPool
        pool = AgentPool()
        pool.register_spec(AgentSpec(
            role=AgentRole.INVESTIGATOR, name="inv",
            tools=["search_logs", "query_metrics"],
        ))
        agent = pool.spawn(AgentRole.INVESTIGATOR)
        mc = MarketCoordinator()
        subtask = Subtask(required_capabilities=["search_logs", "unknown_tool"])
        score = mc._capability_match(agent, subtask)
        assert score == 0.5

    def test_bid_has_fields(self):
        from nil.agents.pool import AgentPool
        pool = AgentPool()
        pool.register_spec(AgentSpec(role=AgentRole.ANALYST, name="anl"))
        agent = pool.spawn(AgentRole.ANALYST)
        mc = MarketCoordinator()
        bid = mc._compute_bid(agent, Subtask(description="test"))
        assert bid.total_cost > 0


# ===========================================================================
# AgentTeam
# ===========================================================================

class TestAgentTeam:
    def test_hierarchical_run(self):
        from nil.agents.pool import AgentPool

        team = AgentTeam(
            leader=AgentSpec(role=AgentRole.ORCHESTRATOR, name="lead"),
            members={
                "investigator": AgentSpec(role=AgentRole.INVESTIGATOR, name="inv"),
                "analyst": AgentSpec(role=AgentRole.ANALYST, name="anl"),
                "executor": AgentSpec(role=AgentRole.EXECUTOR, name="exe"),
                "verifier": AgentSpec(role=AgentRole.VERIFIER, name="ver"),
            },
            coordination="hierarchical",
        )
        result = team.run("Investigate system issue")
        assert isinstance(result.success, bool)
        assert result.output != ""

    def test_swarm_run(self):
        team = AgentTeam(
            leader=AgentSpec(role=AgentRole.ORCHESTRATOR),
            members={
                "a": AgentSpec(role=AgentRole.INVESTIGATOR, name="swarm-a"),
                "b": AgentSpec(role=AgentRole.INVESTIGATOR, name="swarm-b"),
            },
            coordination="swarm",
        )
        result = team.run("Parallel investigation")
        assert result.output != ""

    def test_democratic_run(self):
        team = AgentTeam(
            leader=AgentSpec(role=AgentRole.ORCHESTRATOR),
            members={
                "analyst": AgentSpec(role=AgentRole.ANALYST, name="dem-a"),
                "critic": AgentSpec(role=AgentRole.CRITIC, name="dem-b"),
            },
            coordination="democratic",
        )
        result = team.run("Evaluate proposal")
        assert result.output != ""

    def test_add_subteam(self):
        parent = AgentTeam(name="parent")
        child = AgentTeam(name="child")
        parent.add_subteam(child)
        assert len(parent.subteams) == 1
        assert child.parent is parent

    def test_unknown_coordination_falls_back(self):
        team = AgentTeam(
            leader=AgentSpec(role=AgentRole.ORCHESTRATOR),
            members={
                "inv": AgentSpec(role=AgentRole.INVESTIGATOR),
                "anl": AgentSpec(role=AgentRole.ANALYST),
                "exe": AgentSpec(role=AgentRole.EXECUTOR),
                "ver": AgentSpec(role=AgentRole.VERIFIER),
            },
            coordination="unknown",
        )
        result = team.run("test")
        assert result.success is not None


# ===========================================================================
# MultiAgentNil Integration
# ===========================================================================

class TestMultiAgentNil:
    def test_register_and_run(self):
        society = MultiAgentNil(domain="test")
        society.register_default_specs()
        outcome = society.run("Investigate API error")
        assert isinstance(outcome, MultiAgentOutcome)
        assert outcome.total_duration > 0

    def test_blackboard_is_shared(self):
        society = MultiAgentNil(domain="test")
        society.register_default_specs()
        society.blackboard.write("test.key", "shared_value")
        assert society.blackboard.read("test.key") == "shared_value"

    def test_register_custom_spec(self):
        society = MultiAgentNil(domain="test")
        society.register_agent_spec(AgentSpec(
            role=AgentRole.MONITOR,
            name="watchdog",
            model="claude-haiku-3-20250301",
        ))
        spec = society.agent_pool.get_spec(AgentRole.MONITOR)
        assert spec is not None
        assert spec.name == "watchdog"
