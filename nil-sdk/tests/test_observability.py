from __future__ import annotations

import pytest
import time
import uuid

from nil.observability.events import (
    EventType, ObservabilityEvent, EventRecorder, ObservabilityStore,
)
from nil.observability.trace_viewer import TraceViewer, TimelineView, TimelineRow
from nil.observability.flame_graph import FlameGraphGenerator, FlameGraph, FlameNode
from nil.observability.token_timeline import TokenTimeline, TokenReport, TokenStep
from nil.observability.reasoning_tree import ReasoningTree, ReasoningNode
from nil.observability.memory_browser import MemoryBrowser
from nil.observability.evolution_diff import EvolutionDiffViewer, EvolutionDiff
from nil.observability.strategy_lineage import StrategyLineage, LineageGraph, LineageNode
from nil.observability.debugger import LiveDebugger, DebugCommand, DebugCommandType, Breakpoint, AgentSnapshot


class TestEventType:
    def test_values(self):
        assert EventType.RUN_STARTED.value == "run_started"
        assert EventType.RUN_COMPLETED.value == "run_completed"
        assert EventType.LLM_CALL_STARTED.value == "llm_call_started"


class TestObservabilityEvent:
    def test_defaults(self):
        e = ObservabilityEvent(event_type=EventType.RUN_STARTED, run_id="run-1")
        assert e.event_id != ""
        assert e.timestamp > 0
        assert e.trace_id == "run-1"

    def test_manual_id(self):
        e = ObservabilityEvent(event_id="abc", event_type=EventType.RUN_STARTED)
        assert e.event_id == "abc"

    def test_to_dict_roundtrip(self):
        e = ObservabilityEvent(
            event_type=EventType.LLM_CALL_COMPLETED,
            run_id="run-1",
            payload={"tokens_in": 100, "tokens_out": 50},
            span_id="span-1",
        )
        d = e.to_dict()
        restored = ObservabilityEvent.from_dict(d)
        assert restored.event_type == EventType.LLM_CALL_COMPLETED
        assert restored.run_id == "run-1"
        assert restored.payload["tokens_in"] == 100
        assert restored.span_id == "span-1"


class TestObservabilityStore:
    def test_write_and_get(self):
        store = ObservabilityStore()
        e = ObservabilityEvent(event_type=EventType.RUN_STARTED, run_id="run-1")
        store.write(e)
        assert store.size() == 1
        results = store.get_events(run_id="run-1")
        assert len(results) == 1

    def test_filter_by_type(self):
        store = ObservabilityStore()
        store.write(ObservabilityEvent(event_type=EventType.RUN_STARTED, run_id="r1"))
        store.write(ObservabilityEvent(event_type=EventType.LLM_CALL_STARTED, run_id="r1"))
        results = store.get_events(event_type=EventType.RUN_STARTED)
        assert len(results) == 1

    def test_clear(self):
        store = ObservabilityStore()
        store.write(ObservabilityEvent(event_type=EventType.RUN_STARTED, run_id="r1"))
        store.clear()
        assert store.size() == 0

    def test_to_dict_roundtrip(self):
        store = ObservabilityStore()
        store.write(ObservabilityEvent(event_type=EventType.RUN_STARTED, run_id="r1"))
        d = store.to_dict()
        restored = ObservabilityStore.from_dict(d)
        assert len(restored._events) == 1

    def test_filter_multiple_criteria(self):
        store = ObservabilityStore()
        store.write(ObservabilityEvent(event_type=EventType.RUN_STARTED, run_id="r1", agent_id="a1", domain="d1"))
        store.write(ObservabilityEvent(event_type=EventType.RUN_STARTED, run_id="r2", agent_id="a1", domain="d1"))
        store.write(ObservabilityEvent(event_type=EventType.RUN_STARTED, run_id="r3", agent_id="a2", domain="d2"))
        assert len(store.get_events(agent_id="a1")) == 2
        assert len(store.get_events(agent_id="a1", domain="d1")) == 2
        assert len(store.get_events(agent_id="a2")) == 1


class TestEventRecorder:
    def test_record(self):
        store = ObservabilityStore()
        recorder = EventRecorder(store)
        e = recorder.record(EventType.RUN_STARTED, run_id="run-1")
        assert e.run_id == "run-1"
        assert e.event_id != ""
        assert store.size() == 1

    def test_span_management(self):
        recorder = EventRecorder()
        sid = recorder.start_span("llm_think")
        assert sid.startswith("llm_think-")
        recorder.end_span()
        assert recorder._current_span_stack == []

    def test_record_with_payload(self):
        recorder = EventRecorder()
        e = recorder.record(EventType.LLM_CALL_COMPLETED, run_id="run-1", payload={"tokens": 100})
        assert e.payload["tokens"] == 100

    def test_set_agent_and_domain(self):
        recorder = EventRecorder()
        recorder.agent_id = "agent-1"
        recorder.domain = "incident-response"
        e = recorder.record(EventType.RUN_STARTED, run_id="r1")
        assert e.agent_id == "agent-1"
        assert e.domain == "incident-response"

    def test_to_dict_roundtrip(self):
        recorder = EventRecorder()
        recorder.agent_id = "a1"
        recorder.record(EventType.RUN_STARTED, run_id="r1")
        d = recorder.to_dict()
        restored = EventRecorder.from_dict(d)
        assert restored.agent_id == "a1"
        assert restored.store.size() == 1


class TestTraceViewer:
    def test_empty_run(self):
        store = ObservabilityStore()
        viewer = TraceViewer(store)
        view = viewer.render_timeline("nonexistent")
        assert view.run_id == "nonexistent"

    def test_render_timeline(self):
        store = ObservabilityStore()
        now = time.time()
        store.write(ObservabilityEvent(
            event_type=EventType.RUN_STARTED, run_id="r1",
            timestamp=now,
            span_id="run",
        ))
        store.write(ObservabilityEvent(
            event_type=EventType.LLM_CALL_STARTED, run_id="r1",
            timestamp=now + 0.5,
            span_id="llm",
        ))
        store.write(ObservabilityEvent(
            event_type=EventType.LLM_CALL_COMPLETED, run_id="r1",
            timestamp=now + 1.0,
            span_id="llm",
            payload={"tokens_in": 100, "tokens_out": 50},
        ))
        viewer = TraceViewer(store)
        view = viewer.render_timeline("r1")
        assert view.total_duration > 0
        assert len(view.rows) >= 1


class TestFlameGraphGenerator:
    def test_empty(self):
        store = ObservabilityStore()
        gen = FlameGraphGenerator(store)
        graph = gen.generate("nonexistent")
        assert graph.run_id == "nonexistent"

    def test_generate(self):
        store = ObservabilityStore()
        now = time.time()
        store.write(ObservabilityEvent(
            event_type=EventType.LLM_CALL_STARTED, run_id="r1",
            timestamp=now, span_id="llm1",
        ))
        store.write(ObservabilityEvent(
            event_type=EventType.LLM_CALL_COMPLETED, run_id="r1",
            timestamp=now + 0.5, span_id="llm1",
        ))
        gen = FlameGraphGenerator(store)
        graph = gen.generate("r1")
        assert graph.total_ms > 0
        assert graph.root is not None

    def test_text_rendering(self):
        store = ObservabilityStore()
        gen = FlameGraphGenerator(store)
        text = gen.render_text("r1")
        assert "Flame Graph" in text

    def test_flame_node_defaults(self):
        node = FlameNode(name="test", value=100)
        assert node.name == "test"
        assert node.value == 100
        assert node.children == []


class TestTokenTimeline:
    def test_empty_events(self):
        timeline = TokenTimeline()
        report = timeline.compute([])
        assert report.total_tokens_in == 0
        assert report.total_tokens_out == 0

    def test_compute(self):
        timeline = TokenTimeline()
        events = [
            ObservabilityEvent(
                event_type=EventType.LLM_CALL_COMPLETED, run_id="r1",
                payload={"tokens_in": 100, "tokens_out": 50, "step_number": 1},
            ),
            ObservabilityEvent(
                event_type=EventType.MEMORY_RETRIEVAL, run_id="r1",
                payload={"context_tokens": 200, "step_number": 0, "source": "skills"},
            ),
        ]
        report = timeline.compute(events)
        assert report.total_tokens_in == 300
        assert report.total_tokens_out == 50
        assert report.total_cost > 0

    def test_cost_estimate(self):
        timeline = TokenTimeline()
        cost = timeline._estimate_cost(1000, 500)
        assert cost == round(1000 * 3e-6 + 500 * 15e-6, 6)

    def test_text_rendering(self):
        timeline = TokenTimeline()
        text = timeline.render_text([])
        assert "Token Timeline" in text


class TestReasoningTree:
    def test_capture_and_get(self):
        tree = ReasoningTree()
        node = tree.capture(
            run_id="r1", step_number=1,
            prompt="analyze", response="check logs",
            options=["search_logs", "query_metrics"],
            chosen="search_logs",
            rationale="logs show errors",
            influences=["skill_investigate_timeout"],
        )
        assert node.run_id == "r1"
        assert node.chosen == "search_logs"
        nodes = tree.get_nodes("r1")
        assert len(nodes) == 1

    def test_clear(self):
        tree = ReasoningTree()
        tree.capture(run_id="r1", step_number=1, prompt="p", response="r")
        tree.clear()
        assert tree.get_nodes("r1") == []

    def test_render_text_empty(self):
        tree = ReasoningTree()
        text = tree.render_text("nonexistent")
        assert "No reasoning data" in text

    def test_render_text(self):
        tree = ReasoningTree()
        tree.capture(run_id="r1", step_number=1, prompt="analyze", response="check",
                      options=["tool_a"], chosen="tool_a", rationale="best tool")
        text = tree.render_text("r1")
        assert "Step 1" in text
        assert "tool_a" in text


class TestMemoryBrowser:
    def test_empty(self):
        browser = MemoryBrowser()
        text = browser.render_text("domain")
        assert "No skills" in text

    def test_with_skills(self):
        browser = MemoryBrowser()
        browser.set_skills([
            {"name": "search_tool", "confidence": 0.9, "usage_count": 10, "status": "active"},
        ])
        browser.set_strategies([
            {"version": 1, "is_active": True, "created_at": "now"},
        ])
        browser.set_facts([
            {"content": "API latency is high", "confidence": 0.8},
        ])
        text = browser.render_text("domain")
        assert "search_tool" in text
        assert "API latency" in text

    def test_render_html(self):
        browser = MemoryBrowser()
        html = browser.render_html("domain")
        assert "<html>" in html
        assert "Memory Browser" in html


class TestEvolutionDiffViewer:
    def test_empty(self):
        store = ObservabilityStore()
        viewer = EvolutionDiffViewer(store)
        diff = viewer.render("evol-1")
        assert diff.evolution_id == "evol-1"

    def test_text_rendering(self):
        store = ObservabilityStore()
        viewer = EvolutionDiffViewer(store)
        text = viewer.render_text("evol-1")
        assert "Evolution Run" in text


class TestStrategyLineage:
    def test_empty(self):
        lineage = StrategyLineage()
        text = lineage.render_text("domain")
        assert "No version history" in text

    def test_build_graph(self):
        lineage = StrategyLineage()
        lineage.set_strategies([
            {"version": 1, "mutation_type": "initial", "is_active": False, "created_at": "t1"},
            {"version": 2, "mutation_type": "skill", "parent_version": 1, "is_active": True, "created_at": "t2"},
        ])
        graph = lineage.build("domain")
        assert len(graph.nodes) == 2

    def test_render_text(self):
        lineage = StrategyLineage()
        lineage.set_strategies([
            {"version": 1, "mutation_type": "initial", "is_active": False, "created_at": "t1"},
            {"version": 2, "mutation_type": "skill", "parent_version": 1, "is_active": True, "created_at": "t2"},
        ])
        text = lineage.render_text("domain")
        assert "v1" in text
        assert "v2" in text
        assert "current" in text or "=" in text


class TestLiveDebugger:
    def test_pause_resume(self):
        debugger = LiveDebugger()
        assert debugger.paused is False
        debugger.pause()
        assert debugger.paused is True
        debugger.resume()
        assert debugger.paused is False

    def test_should_pause_on_breakpoint(self):
        debugger = LiveDebugger()
        bp = Breakpoint(tool_name="search_logs")
        debugger.set_breakpoint(bp)
        assert debugger.should_pause(1, tool="search_logs") is True
        assert debugger.should_pause(1, tool="other") is False

    def test_step_breakpoint(self):
        debugger = LiveDebugger()
        bp = Breakpoint(step_number=5)
        debugger.set_breakpoint(bp)
        assert debugger.should_pause(5) is True
        assert debugger.should_pause(3) is False

    def test_take_snapshot(self):
        debugger = LiveDebugger()
        snap = debugger.take_snapshot(run_id="r1", step_number=1, system_prompt="test")
        assert snap.run_id == "r1"
        assert snap.step_number == 1
        assert len(debugger.snapshot_history) == 1

    def test_rollback(self):
        debugger = LiveDebugger()
        debugger.take_snapshot(run_id="r1", step_number=1)
        debugger.take_snapshot(run_id="r1", step_number=2)
        rolled = debugger.rollback()
        assert rolled is not None
        assert rolled.step_number == 1

    def test_rollback_insufficient(self):
        debugger = LiveDebugger()
        assert debugger.rollback() is None

    def test_execute_commands(self):
        debugger = LiveDebugger()
        assert "Resumed" in debugger.execute_command(DebugCommand(type=DebugCommandType.CONTINUE))
        assert "Stepping" in debugger.execute_command(DebugCommand(type=DebugCommandType.STEP))
        assert "Inspecting" in debugger.execute_command(DebugCommand(type=DebugCommandType.INSPECT, payload={"q": "test"}))
        assert "Breakpoint" in debugger.execute_command(DebugCommand(type=DebugCommandType.SET_BREAKPOINT, payload={"condition": "x"}))
        assert "Nothing" in debugger.execute_command(DebugCommand(type=DebugCommandType.ROLLBACK))

    def test_clear_breakpoints(self):
        debugger = LiveDebugger()
        debugger.set_breakpoint(Breakpoint(step_number=5))
        debugger.clear_breakpoints()
        assert debugger.breakpoints == []


class TestIntegration:
    def test_recorder_to_viewer_flow(self):
        store = ObservabilityStore()
        recorder = EventRecorder(store)
        recorder.agent_id = "agent-1"
        recorder.domain = "test"

        run_id = "integration-test"
        recorder.record(EventType.RUN_STARTED, run_id=run_id)
        recorder.record(EventType.LLM_CALL_STARTED, run_id=run_id, payload={"step_number": 1})
        recorder.record(EventType.LLM_CALL_COMPLETED, run_id=run_id, payload={
            "tokens_in": 100, "tokens_out": 50, "cost": 0.001, "step_number": 1,
        })
        recorder.record(EventType.TOOL_CALL_STARTED, run_id=run_id, payload={"tool": "search"})
        recorder.record(EventType.TOOL_CALL_COMPLETED, run_id=run_id, payload={"tool": "search", "duration": 1.2})
        recorder.record(EventType.RUN_COMPLETED, run_id=run_id)

        viewer = TraceViewer(store)
        view = viewer.render_timeline(run_id)
        assert view.total_duration >= 0
        assert len(view.rows) >= 1

        gen = FlameGraphGenerator(store)
        graph = gen.generate(run_id)
        assert graph.total_ms >= 0
        assert graph.root is not None

        timeline = TokenTimeline()
        report = timeline.compute(store.get_events())
        assert report.total_tokens_in >= 100
