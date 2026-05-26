"""Tests for the graph execution engine."""

import time

import pytest

from nil.kernel.graph.graph_types import (
    AgentSpec,
    DependencyEdge,
    EdgeType,
    ExecutionGraph,
    ExecutionNode,
    GraphMetadata,
    GraphOutcome,
    NodeResult,
    NodeStatus,
    NodeType,
)
from nil.kernel.graph.planner import Planner, PlanningStrategy, PlanStep
from nil.kernel.graph.scheduler import GraphScheduler
from nil.kernel.graph.critic import Critic, CritiqueVerdict, SchemaValidator, ErrorFreeValidator
from nil.kernel.graph.templates import GraphTemplate, TemplateLibrary, BUILTIN_TEMPLATES
from nil.kernel.graph.runner import GraphAgentRunner


# ---- Graph Data Model Tests ----


class TestGraphTypes:
    def test_create_execution_node(self):
        node = ExecutionNode(node_id="n1", node_type=NodeType.LLM_CALL, label="test")
        assert node.node_id == "n1"
        assert node.node_type == NodeType.LLM_CALL
        assert node.status == NodeStatus.PENDING

    def test_create_dependency_edge(self):
        edge = DependencyEdge(source="a", target="b", edge_type=EdgeType.DATA_FLOW)
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.edge_type == EdgeType.DATA_FLOW

    def test_agent_spec_defaults(self):
        spec = AgentSpec(model="claude-haiku-3-20250301")
        assert spec.temperature == 0.0
        assert spec.max_tokens == 2048

    def test_node_to_dict_roundtrip(self):
        node = ExecutionNode(
            node_id="n1", node_type=NodeType.TOOL_CALL,
            label="search", task="search logs",
            tool_name="search_logs", max_retries=5,
        )
        d = node.to_dict()
        restored = ExecutionNode.from_dict(d)
        assert restored.node_id == "n1"
        assert restored.node_type == NodeType.TOOL_CALL
        assert restored.tool_name == "search_logs"
        assert restored.max_retries == 5

    def test_subagent_node_type(self):
        from nil.kernel.graph.graph_types import SubagentNode
        sn = SubagentNode(node_id="sub1", sub_domain="nested", sub_task="deep dive")
        assert sn.node_type == NodeType.SUBGRAPH
        assert sn.sub_domain == "nested"

    def test_speculative_node_type(self):
        from nil.kernel.graph.graph_types import SpeculativeNode
        sp = SpeculativeNode(node_id="spec1", max_branches=5)
        assert sp.node_type == NodeType.SPECULATIVE
        assert sp.max_branches == 5


# ---- ExecutionGraph Tests ----


class TestExecutionGraph:
    def test_empty_graph(self):
        g = ExecutionGraph()
        assert len(g.nodes) == 0
        assert len(g.edges) == 0

    def test_add_node_and_edge(self):
        g = ExecutionGraph()
        n1 = ExecutionNode(node_id="a")
        n2 = ExecutionNode(node_id="b")
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(DependencyEdge(source="a", target="b"))
        assert "a" in g.nodes
        assert g.get_dependents("a") == ["b"]
        assert g.get_dependencies("b") == ["a"]

    def test_metadata_defaults(self):
        g = ExecutionGraph(metadata=GraphMetadata(domain="test", task="do something"))
        assert g.metadata.domain == "test"
        assert g.metadata.task == "do something"
        assert g.metadata.max_concurrency == 5

    def test_node_output_storage(self):
        g = ExecutionGraph()
        g.set_node_output("n1", {"result": "ok"})
        assert g.get_node_output("n1") == {"result": "ok"}

    def test_find_ready_nodes_simple(self):
        g = ExecutionGraph()
        n1 = ExecutionNode(node_id="a")
        n2 = ExecutionNode(node_id="b")
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(DependencyEdge(source="a", target="b"))

        ready = g.find_ready_nodes()
        assert len(ready) == 1
        assert ready[0].node_id == "a"

        g.mark_completed("a")
        ready = g.find_ready_nodes()
        assert len(ready) == 1
        assert ready[0].node_id == "b"

    def test_find_ready_parallel_nodes(self):
        g = ExecutionGraph()
        g.add_node(ExecutionNode(node_id="input"))
        g.add_node(ExecutionNode(node_id="a"))
        g.add_node(ExecutionNode(node_id="b"))
        g.add_node(ExecutionNode(node_id="c"))
        g.add_node(ExecutionNode(node_id="merge"))
        for nid in ["a", "b", "c"]:
            g.add_edge(DependencyEdge(source="input", target=nid))
        g.add_edge(DependencyEdge(source="a", target="merge"))
        g.add_edge(DependencyEdge(source="b", target="merge"))
        g.add_edge(DependencyEdge(source="c", target="merge"))

        g.mark_completed("input")
        ready = g.find_ready_nodes(max_concurrency=10)
        assert len(ready) == 3  # a, b, c all ready in parallel

    def test_find_ready_respects_concurrency(self):
        g = ExecutionGraph()
        g.add_node(ExecutionNode(node_id="input"))
        for i in range(10):
            g.add_node(ExecutionNode(node_id=f"n{i}"))
            g.add_edge(DependencyEdge(source="input", target=f"n{i}"))
        g.mark_completed("input")

        ready = g.find_ready_nodes(max_concurrency=3)
        assert len(ready) <= 3

    def test_all_completed(self):
        g = ExecutionGraph()
        g.add_node(ExecutionNode(node_id="a"))
        g.add_node(ExecutionNode(node_id="b"))
        assert not g.all_completed()
        g.mark_completed("a")
        g.mark_completed("b")
        assert g.all_completed()

    def test_fallback_activation(self):
        g = ExecutionGraph()
        g.add_node(ExecutionNode(node_id="primary"))
        g.add_node(ExecutionNode(node_id="fallback"))
        g.activate_fallback("primary", "fallback")
        assert g.get_node("primary").status == NodeStatus.FAILED
        assert g.get_node("fallback").status == NodeStatus.READY

    def test_rollback_activation(self):
        g = ExecutionGraph()
        g.add_node(ExecutionNode(node_id="primary"))
        g.add_node(ExecutionNode(node_id="rollback"))
        g.activate_rollback("primary", "rollback")
        assert g.get_node("primary").status == NodeStatus.ROLLING_BACK
        assert g.get_node("rollback").status == NodeStatus.READY

    def test_to_dict_roundtrip(self):
        g = ExecutionGraph(metadata=GraphMetadata(domain="d", task="t"))
        g.add_node(ExecutionNode(node_id="a"))
        g.add_node(ExecutionNode(node_id="b"))
        g.add_edge(DependencyEdge(source="a", target="b"))
        d = g.to_dict()
        restored = ExecutionGraph.from_dict(d)
        assert "a" in restored.nodes
        assert "b" in restored.nodes
        assert len(restored.edges) == 1

    def test_conditional_edge_blocks_when_condition_false(self):
        g = ExecutionGraph()
        g.add_node(ExecutionNode(node_id="check"))
        g.add_node(ExecutionNode(node_id="proceed"))
        g.add_edge(DependencyEdge(source="check", target="proceed",
                                   edge_type=EdgeType.CONDITIONAL, condition="pass"))
        g.mark_completed("check")
        g.set_node_output("check", {"verdict": "fail"})
        ready = g.find_ready_nodes()
        assert len(ready) == 0  # condition not met

    def test_get_upstream_output(self):
        g = ExecutionGraph()
        g.add_node(ExecutionNode(node_id="a"))
        g.add_node(ExecutionNode(node_id="b"))
        g.add_edge(DependencyEdge(source="a", target="b"))
        g.set_node_output("a", {"key": "value"})
        output = g.get_upstream_output("b", field="key")
        assert output == "value"


# ---- Scheduler Tests ----


class TestGraphScheduler:
    def test_scheduler_sequential_execution(self):
        scheduler = GraphScheduler(max_concurrency=1)
        g = ExecutionGraph()
        results = {}

        def make_executor(node_id: str, value: str):
            def executor(node, graph):
                results[node.node_id] = value
                return NodeResult(node_id=node.node_id, success=True, output=value)
            return executor

        scheduler.register_executor(NodeType.LLM_CALL, make_executor("a", "step_a"))
        scheduler.register_executor(NodeType.TOOL_CALL, make_executor("b", "step_b"))

        g.add_node(ExecutionNode(node_id="a", node_type=NodeType.LLM_CALL))
        g.add_node(ExecutionNode(node_id="b", node_type=NodeType.TOOL_CALL))
        g.add_edge(DependencyEdge(source="a", target="b"))

        outcome = scheduler.execute(g)
        assert outcome.success
        assert "a" in outcome.node_results
        assert "b" in outcome.node_results

    def test_scheduler_parallel_execution(self):
        scheduler = GraphScheduler(max_concurrency=10)
        g = ExecutionGraph()
        timestamps: dict[str, float] = {}

        def make_timestamp_executor(node_id: str):
            def executor(node, graph):
                timestamps[node.node_id] = time.time()
                return NodeResult(node_id=node.node_id, success=True, output="done")
            return executor

        scheduler.register_executor(NodeType.TOOL_CALL, make_timestamp_executor(""))

        g.add_node(ExecutionNode(node_id="input", node_type=NodeType.INPUT))
        for i in range(5):
            n = ExecutionNode(node_id=f"p{i}", node_type=NodeType.TOOL_CALL)
            g.add_node(n)
            g.add_edge(DependencyEdge(source="input", target=f"p{i}"))
        g.mark_completed("input")

        scheduler.unregister_executor(NodeType.TOOL_CALL)

        def fast_exec(node, graph):
            return NodeResult(node_id=node.node_id, success=True, output="done")

        scheduler.register_executor(NodeType.TOOL_CALL, fast_exec)
        outcome = scheduler.execute(g)
        assert outcome.success
        assert outcome.node_count == 5

    def test_scheduler_node_failure_no_fallback(self):
        scheduler = GraphScheduler(max_concurrency=1)

        def failing_executor(node, graph):
            return NodeResult(node_id=node.node_id, success=False, error="intentional failure")

        scheduler.register_executor(NodeType.LLM_CALL, failing_executor)
        g = ExecutionGraph()
        g.add_node(ExecutionNode(node_id="a", node_type=NodeType.LLM_CALL))
        g.add_node(ExecutionNode(node_id="b", node_type=NodeType.LLM_CALL))
        g.add_edge(DependencyEdge(source="a", target="b"))
        g.mark_completed("a")
        outcome = scheduler.execute(g)
        assert not outcome.success

    def test_scheduler_node_failure_with_fallback(self):
        scheduler = GraphScheduler(max_concurrency=1)

        def make_result(node_id: str, success: bool):
            def executor(node, graph):
                return NodeResult(node_id=node.node_id, success=success, output="done")
            return executor

        scheduler.register_executor(NodeType.LLM_CALL, make_result("", False))
        scheduler.unregister_executor(NodeType.LLM_CALL)

        def primary_exec(node, graph):
            return NodeResult(node_id=node.node_id, success=False, error="primary failed")

        def fallback_exec(node, graph):
            return NodeResult(node_id=node.node_id, success=True, output="fallback succeeded")

        scheduler.register_executor(NodeType.LLM_CALL, primary_exec)
        g = ExecutionGraph()
        g.add_node(ExecutionNode(node_id="primary", node_type=NodeType.LLM_CALL,
                                  fallback_node="fallback"))
        g.add_node(ExecutionNode(node_id="fallback", node_type=NodeType.LLM_CALL))

        r = g.find_ready_nodes()
        outcome = scheduler.execute(g)
        assert outcome.success

    def test_scheduler_cost_budget(self):
        scheduler = GraphScheduler(max_concurrency=10, max_cost_per_run=0.0)
        g = ExecutionGraph()

        def free_executor(node, graph):
            return NodeResult(node_id=node.node_id, success=True, output="ok")

        scheduler.register_executor(NodeType.LLM_CALL, free_executor)
        g.metadata.total_cost_budget = 0.0
        for i in range(3):
            g.add_node(ExecutionNode(node_id=f"n{i}", node_type=NodeType.LLM_CALL))
        outcome = scheduler.execute(g)
        assert outcome.success

    def test_scheduler_merge_node(self):
        scheduler = GraphScheduler(max_concurrency=5)
        g = ExecutionGraph()

        def data_executor(node, graph):
            return NodeResult(node_id=node.node_id, success=True,
                              output={node.node_id: "data"})

        scheduler.register_executor(NodeType.TOOL_CALL, data_executor)
        scheduler.register_executor(NodeType.MERGE, scheduler._execute_merge)

        g.add_node(ExecutionNode(node_id="a", node_type=NodeType.TOOL_CALL))
        g.add_node(ExecutionNode(node_id="b", node_type=NodeType.TOOL_CALL))
        g.add_node(ExecutionNode(node_id="merge", node_type=NodeType.MERGE))
        g.add_edge(DependencyEdge(source="a", target="merge"))
        g.add_edge(DependencyEdge(source="b", target="merge"))
        g.mark_completed("a")
        g.mark_completed("b")
        g.set_node_output("a", {"from_a": "val_a"})
        g.set_node_output("b", {"from_b": "val_b"})

        result = scheduler._execute_merge(g.get_node("merge"), g)
        assert result.success
        assert result.output.get("from_a") == "val_a"
        assert result.output.get("from_b") == "val_b"


# ---- Planner Tests ----


class TestPlanner:
    def test_planner_monolithic(self):
        planner = Planner()
        graph = planner.plan("do a simple thing", strategy=PlanningStrategy.MONOLITHIC)
        assert graph.metadata.execution_plan == "monolithic"
        assert len(graph.nodes) >= 2

    def test_planner_sequential(self):
        planner = Planner()
        graph = planner.plan("do steps", tools=["tool_a", "tool_b", "tool_c"],
                             strategy=PlanningStrategy.SEQUENTIAL)
        assert graph.metadata.execution_plan == "sequential"
        tool_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.TOOL_CALL]
        assert len(tool_nodes) == 3

    def test_planner_parallel_fan_out(self):
        planner = Planner()
        graph = planner.plan("check a and b and c", tools=["t1", "t2", "t3"],
                             strategy=PlanningStrategy.PARALLEL_FAN_OUT)
        assert graph.metadata.execution_plan == "parallel_fan_out"
        merge_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.MERGE]
        assert len(merge_nodes) >= 1

    def test_planner_hypothesis_testing(self):
        planner = Planner()
        graph = planner.plan("find root cause of outage",
                             strategy=PlanningStrategy.HYPOTHESIS_TESTING,
                             context={"hypotheses": ["check db", "check logs", "check network"]})
        assert graph.metadata.execution_plan == "hypothesis_testing"
        hyp_nodes = [n for n in graph.nodes.values() if n.label and n.label.startswith("Hypothesis")]
        assert len(hyp_nodes) == 3

    def test_planner_contingency(self):
        planner = Planner()
        graph = planner.plan("deploy to production", tools=["precheck", "deploy_tool", "verify"],
                             strategy=PlanningStrategy.CONTINGENCY)
        assert graph.metadata.execution_plan == "contingency"
        rollback_nodes = [n for n in graph.nodes.values() if n.label == "Rollback"]
        assert len(rollback_nodes) >= 1

    def test_planner_explore_exploit(self):
        planner = Planner()
        graph = planner.plan("investigate and fix the error",
                             tools=["search_logs", "query_metrics", "apply_fix"],
                             strategy=PlanningStrategy.EXPLORE_EXPLOIT)
        assert graph.metadata.execution_plan == "explore_exploit"
        critic_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.CRITIC]
        assert len(critic_nodes) >= 1

    def test_infer_strategy(self):
        planner = Planner()
        s1 = planner._infer_strategy("find why the API is slow", ["logs", "metrics"])
        assert s1 == PlanningStrategy.HYPOTHESIS_TESTING

        s2 = planner._infer_strategy("deploy the new version", ["build", "deploy"])
        assert s2 == PlanningStrategy.CONTINGENCY

        s3 = planner._infer_strategy("compare a and b", ["tool_a", "tool_b"])
        assert s3 == PlanningStrategy.PARALLEL_FAN_OUT

    def test_plan_from_template(self):
        planner = Planner()
        planner.register_template("test_template", [
            PlanStep(label="step1", description="First step", node_type=NodeType.TOOL_CALL),
            PlanStep(label="step2", description="Second step", node_type=NodeType.LLM_CALL,
                     depends_on=["step1"]),
        ])
        graph = planner.plan_from_template("test_template", "do task", "domain", ["tool"])
        assert graph is not None
        node_labels = [n.label for n in graph.nodes.values()]
        assert "step1" in node_labels
        assert "step2" in node_labels

    def test_plan_with_llm_fallback(self):
        planner = Planner()
        graph = planner.plan_with_llm("simple task", "domain", ["tool"])
        assert graph is not None  # should fall back gracefully


# ---- Critic Tests ----


class TestCritic:
    def test_critic_pass_on_empty_output(self):
        critic = Critic()
        node = ExecutionNode(node_id="n1")
        result = critic.evaluate_output("valid output", node_id="n1")
        assert result.verdict == CritiqueVerdict.PASS

    def test_critic_fail_on_none_output(self):
        critic = Critic()
        node = ExecutionNode(node_id="n1")
        result = critic.evaluate_output(None, node_id="n1")
        assert result.verdict == CritiqueVerdict.NEEDS_REVISION

    def test_schema_validator_missing_field(self):
        validator = SchemaValidator()
        node = ExecutionNode(node_id="n1", output_schema={"result": "str", "status": "str"})
        issues = validator.check(node, {"result": "ok"}, {})
        assert len(issues) == 1
        assert "status" in issues[0].description

    def test_schema_validator_no_schema_skips(self):
        validator = SchemaValidator()
        node = ExecutionNode(node_id="n1")
        issues = validator.check(node, {"result": "ok"}, {})
        assert len(issues) == 0

    def test_error_free_validator_finds_errors(self):
        validator = ErrorFreeValidator()
        node = ExecutionNode(node_id="n1")
        issues = validator.check(node, {"error": "something went wrong"}, {})
        assert len(issues) >= 1

    def test_critic_score_affects_confidence(self):
        critic = Critic()
        node = ExecutionNode(node_id="n1")
        result = critic.evaluate_output(None)
        assert result.confidence < 1.0

    def test_critic_multiple_validators(self):
        critic = Critic()
        critic.add_validator(SchemaValidator())
        critic.add_validator(ErrorFreeValidator())
        node = ExecutionNode(node_id="n1", output_schema={"result": "str"})
        result = critic.evaluate(node, {"result": "ok"})
        assert result.verdict == CritiqueVerdict.PASS

        result2 = critic.evaluate(node, {"error": "fail"})
        assert result2.verdict in (CritiqueVerdict.NEEDS_REVISION, CritiqueVerdict.FAIL)


# ---- Graph Template Tests ----


class TestGraphTemplates:
    def test_template_library_has_builtins(self):
        lib = TemplateLibrary()
        templates = lib.list_templates()
        names = [t.name for t in templates]
        assert "investigate_and_fix" in names
        assert "root_cause_analysis" in names
        assert "safe_deploy" in names
        assert "incident_triage" in names
        assert "data_migration" in names

    def test_find_template_for_task(self):
        lib = TemplateLibrary()
        matches = lib.find_for_task("fix the broken deployment")
        assert len(matches) >= 1
        assert matches[0].name in ("safe_deploy", "investigate_and_fix")

    def test_template_instantiation(self):
        lib = TemplateLibrary()
        template = lib.get("investigate_and_fix")
        assert template is not None
        graph = template.instantiate("fix api", "incident", ["search", "deploy"])
        assert graph is not None
        assert graph.metadata.execution_plan == "template:investigate_and_fix"
        assert len(graph.nodes) >= 4

    def test_template_record_usage(self):
        template = GraphTemplate(name="test_tpl", steps=[])
        assert template.success_rate == 0.0
        template.record_usage(success=True, cost=0.5, duration_ms=1000)
        assert template.usage_count == 1
        assert template.success_count == 1
        assert template.success_rate == 1.0
        template.record_usage(success=False, cost=0.3, duration_ms=500)
        assert template.usage_count == 2
        assert template.success_rate == 0.5

    def test_template_roundtrip(self):
        lib = TemplateLibrary()
        original = lib.get("incident_triage")
        d = original.to_dict()
        assert d["name"] == "incident_triage"
        assert d["usage_count"] >= 0


# ---- Planner Strategy Inference ----


class TestStrategyInference:
    def test_strategy_inference_priority(self):
        planner = Planner()
        assert planner._infer_strategy("deploy the app", ["build", "test", "deploy"]) == PlanningStrategy.CONTINGENCY
        assert planner._infer_strategy("diagnose failure", ["logs", "metrics"]) == PlanningStrategy.HYPOTHESIS_TESTING
        assert planner._infer_strategy("compare a and b", ["x", "y"]) == PlanningStrategy.PARALLEL_FAN_OUT
        assert planner._infer_strategy("simple", ["t1", "t2", "t3"]) == PlanningStrategy.EXPLORE_EXPLOIT
        assert planner._infer_strategy("simple", []) == PlanningStrategy.MONOLITHIC


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
