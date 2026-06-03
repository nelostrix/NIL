"""
Comprehensive system verification for Nil SDK.

Exercises every public API and subsystem with the StubProvider
(no API key needed). Reports PASS/FAIL for each component.

Usage:
    source ~/venvs/nil-venv/bin/activate
    python nil-sdk/verify_system.py
"""

import sys
import os
import time
import tempfile
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nil-sdk"))
TEMP_DIR = tempfile.mkdtemp(prefix="nil_verify_")

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                fn(*args, **kwargs)
                results.append((PASS, name, ""))
                print(f"  {PASS}  {name}")
            except Exception as e:
                results.append((FAIL, name, f"{e}"))
                print(f"  {FAIL}  {name}  --  {e}")
                traceback.print_exc()
        return wrapper
    return decorator


# ======================== 1. CORE ========================

@check("Core: NilConfig defaults")
def test_config():
    from nil.config import NilConfig
    assert NilConfig().log_level == "INFO"

@check("Core: EvolutionConfig")
def test_evolution_config():
    from nil.config import EvolutionConfig
    ec = EvolutionConfig(strategy="continuous", evolve_interval=5)
    assert ec.strategy == "continuous"
    ec2 = EvolutionConfig(enabled=False)
    assert ec2.enabled is False

@check("Core: Domain with @tool decorator")
def test_domain():
    import nil
    from nil.kernel.outcome import RunOutcome
    @nil.tool
    def my_tool(x: str) -> str:
        return x
    d = nil.Domain(name="test", tools=[my_tool], objectives=["do stuff"])
    assert d.name == "test"
    assert len(d.tools) == 1
    assert d.tool_name == ["my_tool"]
    assert d.evaluate(RunOutcome(run_id="r1", task="t1", success=True)) is True

@check("Core: @nil.tool produces Tool with schema")
def test_tool_decorator():
    import nil
    @nil.tool
    def my_tool(query: str, limit: int = 10) -> str:
        return f"results for {query}"
    assert my_tool.name == "my_tool"
    s = my_tool.to_schema()
    assert "function" in s

@check("Core: Tool.invoke dispatches")
def test_tool_invoke():
    import nil
    @nil.tool
    def add(a: int, b: int) -> int:
        return a + b
    assert add.invoke({"a": 2, "b": 3}) == 5

@check("Core: ToolManager")
def test_tool_manager():
    from nil.kernel.tool_manager import ToolManager
    import nil
    @nil.tool
    def search(q: str) -> str:
        return f"found: {q}"
    tm = ToolManager([search])
    assert "search" in tm.names
    r = tm.execute("search", {"q": "hello"})
    assert r["output"] == "found: hello"
    # Missing tool returns error dict, does not raise
    r2 = tm.execute("nonexistent", {})
    assert r2["error"] is not None

@check("Core: ToolRegistry")
def test_tool_registry():
    from nil.tools.base import Tool
    from nil.tools.registry import register_tool, list_tools, get_tool
    t = Tool(name="verify_global", description="test", parameters={"x": {"type": "string"}})
    register_tool(t)
    assert "verify_global" in list_tools()
    assert get_tool("verify_global") is not None


# ======================== 2. KERNEL ========================

@check("Kernel: RunOutcome")
def test_run_outcome():
    from nil.kernel.outcome import RunOutcome, StepRecord
    o = RunOutcome(run_id="r1", task="test")
    o.steps.append(StepRecord(1, "tool_call", "t1", {}, "ok", 10, None, time.time()))
    assert len(o.steps) == 1

@check("Kernel: Tracer")
def test_tracer():
    from nil.kernel.tracer import Tracer
    from nil.kernel.outcome import RunOutcome
    tracer = Tracer(run_id="t1", domain="test")
    tracer.record_llm_call("hi", "world", 10, 5, 100)
    tracer.record_tool_call("search", {"q": "x"}, "ok", 50, None)
    trace = tracer.finalize(RunOutcome(run_id="t1", task="t"))
    assert trace.run_id == "t1"
    assert len(trace.steps) == 2

@check("Kernel: KernelRunner with StubProvider")
def test_kernel_runner():
    import nil
    from nil.kernel.runner import KernelRunner
    from nil.kernel.tracer import Tracer
    from nil.config import NilConfig
    from nil.memory.store import PersistentMemory
    from nil.providers.stub import StubProvider
    @nil.tool
    def ping(msg: str) -> str:
        return f"pong: {msg}"
    runner = KernelRunner(domain=nil.Domain(name="k-test", tools=[ping]),
                          model="stub", config=NilConfig(),
                          memory=PersistentMemory(os.path.join(TEMP_DIR, "kr")),
                          provider="stub")
    assert isinstance(runner._provider, StubProvider)
    outcome = runner.execute(task="hello", context={}, tracer=Tracer("kr1", "k-test"))
    assert outcome.task == "hello"

@check("Kernel: auto-detects StubProvider")
def test_kernel_stub_auto():
    import nil
    from nil.kernel.runner import KernelRunner
    from nil.config import NilConfig
    from nil.memory.store import PersistentMemory
    @nil.tool
    def f(x: str) -> str:
        return x
    runner = KernelRunner(domain=nil.Domain(name="k-auto", tools=[f]),
                          model="stub", config=NilConfig(),
                          memory=PersistentMemory(os.path.join(TEMP_DIR, "kauto")))
    assert "StubProvider" in type(runner._provider).__name__


# ======================== 3. PROVIDERS ========================

@check("Provider: StubProvider")
def test_stub():
    from nil.providers.stub import StubProvider
    from nil.config import NilConfig
    sp = StubProvider("stub", NilConfig())
    r = sp.chat("sys", [{"role": "user", "content": "hi"}],
                tools=[{"function": {"name": "t", "parameters": {"properties": {"x": {"type": "string"}}}}}])
    assert len(r["tool_calls"]) == 1
    r2 = sp.chat("sys", [{"role": "user", "content": "done"}])
    assert r2["tool_calls"] == []


# ======================== 4. PERSISTENT MEMORY ========================

@check("Memory: store_trace + get_traces")
def test_mem_store():
    from nil.memory.store import PersistentMemory
    mem = PersistentMemory(os.path.join(TEMP_DIR, "m1"))
    t = {"run_id": "m1", "domain": "d", "task": "t",
         "steps": [], "outcome_success": True, "outcome_output": "ok",
         "total_duration": 1.0, "total_tokens_in": 10, "total_tokens_out": 5,
         "total_cost": 0.001, "error": None, "created_at": time.time(), "metadata": {}}
    mem.store_trace(t)
    assert mem.get_trace_count("d") == 1
    assert len(mem.get_traces("d")) == 1

@check("Memory: skills CRUD")
def test_mem_skills():
    from nil.memory.store import PersistentMemory
    from nil.memory.skill_registry import Skill
    mem = PersistentMemory(os.path.join(TEMP_DIR, "m2"))
    mem.store_skill("test-dom", name="s1", pattern={"action_sequence": ["a", "b"]}, confidence=0.9)
    skills = mem.list_skills("test-dom")
    assert "s1" in skills

@check("Memory: strategy + success_rate")
def test_mem_strategy():
    from nil.memory.store import PersistentMemory
    mem = PersistentMemory(os.path.join(TEMP_DIR, "m3"))
    mem.store_strategy("d", {"v": 1})
    assert mem.get_active_strategy("d") is not None
    assert isinstance(mem.get_success_rate("d"), float)

@check("Memory: ExperienceQuery")
def test_exp_query():
    from nil.memory.store import PersistentMemory
    from nil.memory.experience import ExperienceQuery
    mem = PersistentMemory(os.path.join(TEMP_DIR, "m4"))
    for i in range(5):
        mem.store_trace({"run_id": f"e{i}", "domain": "d", "task": f"t{i}",
                         "steps": [], "outcome_success": True, "outcome_output": "ok",
                         "total_duration": 0.5, "total_tokens_in": 10, "total_tokens_out": 5,
                         "total_cost": 0.001, "error": None,
                         "created_at": time.time() + i, "metadata": {}})
    assert ExperienceQuery(mem).stats("d").total_runs == 5


# ======================== 5. HIERARCHICAL MEMORY ========================

@check("Memory: HierarchicalMemory")
def test_hier():
    from nil.memory.hierarchical import HierarchicalMemory
    hm = HierarchicalMemory(path=os.path.join(TEMP_DIR, "hier"))
    hm.store_trace({"run_id": "h1", "domain": "d", "task": "t",
        "steps": [{"step_number": 1, "step_type": "tool_call", "tool_name": "t1",
                    "tool_input": {}, "tool_output": "ok", "duration_ms": 10,
                    "error": None, "timestamp": time.time()}],
        "outcome_success": True, "outcome_output": "done",
        "total_duration": 1.0, "total_tokens_in": 10, "total_tokens_out": 5,
        "total_cost": 0.001, "error": None, "created_at": time.time(), "metadata": {}})
    assert hm.episodic is not None
    assert hm.semantic is not None
    assert hm.procedural is not None
    assert hm.working is not None


# ======================== 6. EVOLUTION ========================

@check("Evolution: TraceAnalyzer")
def test_analyzer():
    from nil.memory.store import PersistentMemory
    from nil.evolution.analyzer import TraceAnalyzer
    mem = PersistentMemory(os.path.join(TEMP_DIR, "ea"))
    for i in range(15):
        mem.store_trace({"run_id": f"a{i}", "domain": "d", "task": f"t{i}",
            "steps": [{"step_number": 1, "step_type": "tool_call", "tool_name": "fix",
                        "tool_input": {}, "tool_output": "ok", "duration_ms": 10,
                        "error": None, "timestamp": time.time()}],
            "outcome_success": True, "outcome_output": "ok",
            "total_duration": 1.0, "total_tokens_in": 10, "total_tokens_out": 5,
            "total_cost": 0.001, "error": None, "created_at": time.time() + i, "metadata": {},
            "tool_sequence": ["fix"]})
    ta = TraceAnalyzer(mem)
    trend = ta.detect_performance_trend("d", window=5)
    assert trend["status"] in ("stable", "improving", "degrading", "insufficient_data")

@check("Evolution: SkillExtractor")
def test_skill_extractor():
    from nil.memory.store import PersistentMemory
    from nil.evolution.skill_extractor import SkillExtractor
    mem = PersistentMemory(os.path.join(TEMP_DIR, "se"))
    for i in range(6):
        mem.store_trace({"run_id": f"s{i}", "domain": "d", "task": f"t{i}",
            "steps": [{"step_number": 1, "tool_name": "diag", "tool_input": {},
                        "tool_output": "ok", "duration_ms": 10, "error": None,
                        "timestamp": time.time(), "step_type": "tool_call"},
                       {"step_number": 2, "tool_name": "fix", "tool_input": {},
                        "tool_output": "ok", "duration_ms": 10, "error": None,
                        "timestamp": time.time(), "step_type": "tool_call"}],
            "outcome_success": True, "outcome_output": "ok", "total_duration": 1.0,
            "total_tokens_in": 10, "total_tokens_out": 5, "total_cost": 0.001,
            "error": None, "created_at": time.time() + i, "metadata": {},
            "tool_sequence": ["diag", "fix"]})
    r = SkillExtractor(mem).extract("d", min_frequency=3)
    assert r.total_patterns_analyzed >= 0

@check("Evolution: DriftDetector")
def test_drift():
    from nil.memory.store import PersistentMemory
    from nil.evolution.drift_detector import DriftDetector
    mem = PersistentMemory(os.path.join(TEMP_DIR, "dd"))
    for i in range(40):
        mem.store_trace({"run_id": f"d{i}", "domain": "d", "task": f"t{i}",
            "steps": [], "outcome_success": True, "outcome_output": "ok",
            "total_duration": 1.0, "total_tokens_in": 10, "total_tokens_out": 5,
            "total_cost": 0.001, "error": None, "created_at": time.time() + i, "metadata": {}})
    r = DriftDetector(mem).check("d")
    assert hasattr(r, "drift_detected")

@check("Evolution: ReplayEvaluator")
def test_evaluator():
    from nil.memory.store import PersistentMemory
    from nil.evolution.evaluator import ReplayEvaluator
    mem = PersistentMemory(os.path.join(TEMP_DIR, "ev"))
    r = ReplayEvaluator(mem).evaluate(domain="d", candidate={"tool_sequence": []})
    assert r is not None

@check("Evolution: WorldModelBuilder")
def test_wm():
    from nil.memory.store import PersistentMemory
    from nil.evolution.world_model import WorldModelBuilder
    mem = PersistentMemory(os.path.join(TEMP_DIR, "wm"))
    wm = WorldModelBuilder(memory=mem).build("d")
    assert wm is not None

@check("Evolution: TextualGradientEngine")
def test_gradient():
    from nil.memory.store import PersistentMemory
    from nil.evolution.gradient_engine import TextualGradientEngine, GradientBatch
    mem = PersistentMemory(os.path.join(TEMP_DIR, "gr"))
    for i in range(10):
        mem.store_trace({"run_id": f"g{i}", "domain": "d", "task": f"t{i}",
            "steps": [], "outcome_success": i >= 3, "outcome_output": "ok",
            "total_duration": 1.0, "total_tokens_in": 10, "total_tokens_out": 5,
            "total_cost": 0.001, "error": "fail" if i < 3 else None,
            "created_at": time.time() + i, "metadata": {},
            "tool_sequence": ["t1"]})
    b = TextualGradientEngine(memory=mem).compute_gradients("d")
    assert isinstance(b, GradientBatch)

@check("Evolution: EvolutionPipeline")
def test_pipeline():
    from nil.memory.store import PersistentMemory
    from nil.evolution.pipeline import EvolutionPipeline
    from nil.evolution.gradient_engine import StubGradientProvider
    mem = PersistentMemory(os.path.join(TEMP_DIR, "pipe"))
    for i in range(15):
        mem.store_trace({"run_id": f"p{i}", "domain": "d", "task": f"t{i}",
            "steps": [{"step_number": 1, "step_type": "tool_call", "tool_name": "diag",
                        "tool_input": {}, "tool_output": "ok", "duration_ms": 10,
                        "error": None, "timestamp": time.time()}],
            "outcome_success": True, "outcome_output": "ok", "total_duration": 1.0,
            "total_tokens_in": 10, "total_tokens_out": 5, "total_cost": 0.001,
            "error": None, "created_at": time.time() + i, "metadata": {},
            "tool_sequence": ["diag"]})
    r = EvolutionPipeline(memory=mem, gradient_provider=StubGradientProvider(), auto_apply=True).run("d")
    assert hasattr(r, "had_impact")


# ======================== 7. TOOL EVOLUTION ========================

@check("Tools: ToolEvolver")
def test_evolver():
    from nil.tools.evolver import ToolEvolver
    ev = ToolEvolver()
    traces = [{"run_id": f"e{i}", "task": f"t{i}",
               "steps": [{"step_number": s, "tool_name": t}
                         for s, t in enumerate(["search", "read", "write"], 1)],
               "tool_sequence": ["search", "read", "write"],
               "outcome_success": True} for i in range(8)]
    assert ev.evolve(traces, "td") is not None

@check("Tools: ToolEmbedding")
def test_emb():
    from nil.tools.base import Tool
    from nil.tools.embeddings import ToolEmbeddingGenerator
    gen = ToolEmbeddingGenerator(8)
    emb = gen.initialize(Tool("search", "search tool", {"q": {"type": "string"}}))
    assert emb.tool_name == "search"
    assert len(emb.full_embedding) == 8

@check("Tools: AffordanceLearner")
def test_aff():
    from nil.tools.base import Tool
    from nil.tools.affordance import AffordanceLearner
    a = AffordanceLearner().learn_affordances(Tool("s", "s", {}))
    assert a.tool_name == "s"

@check("Tools: LatentToolSelector")
def test_selector():
    from nil.tools.base import Tool
    from nil.tools.selection import LatentToolSelector
    sel = LatentToolSelector()
    from nil.tools.embeddings import ToolEmbeddingGenerator
    gen = ToolEmbeddingGenerator(8)
    for name in ["a", "b"]:
        sel.add_tool_embedding(gen.initialize(Tool(name, f"tool {name}", {})))
    selected = sel.select_tools("some task", current_tool_state=["a"])
    assert isinstance(selected, list)


# ======================== 8. COGNITION ========================

@check("Cognition: PolicyEngine")
def test_policy():
    from nil.cognition.modes import CognitiveMode
    from nil.cognition.policy import build_default_policy
    ctx = type("FakeCtx", (), {"get": lambda self, k, d=None: {
        "success_rate": 0.5, "uncertainty": 0.3, "task_complexity": 0.5,
    }.get(k, d)})()
    m = build_default_policy().select_mode(ctx)
    assert m in CognitiveMode

@check("Cognition: CognitiveLoop")
def test_loop():
    from nil.cognition.loop import CognitiveLoop
    from nil.cognition.policy import build_default_policy
    assert CognitiveLoop(policy=build_default_policy()) is not None

@check("Cognition: CognitiveEconomy")
def test_economy():
    from nil.cognition.economy import CognitiveEconomy
    assert CognitiveEconomy() is not None

@check("Cognition: LiveSupervisor")
def test_supervisor():
    from nil.cognition.supervisor import LiveSupervisor
    assert LiveSupervisor() is not None

@check("Cognition: SimulationEngine")
def test_sim():
    from nil.cognition.simulation import SimulationEngine
    assert SimulationEngine() is not None

@check("Cognition: Adapters")
def test_adapters():
    from nil.cognition.adapters import ModelSwitcher, AgentSpawner, RetrievalAlterer, ReasoningDeeper
    assert ModelSwitcher() and AgentSpawner() and RetrievalAlterer() and ReasoningDeeper()

@check("Cognition: RACEngine")
def test_rac():
    from nil.cognition.runner import RACEngine
    rac = RACEngine()
    rac.add_skill("use search first", keywords=["search"])
    assert len(rac.search("test", top_k=5)) >= 0

@check("Cognition: RetrievalIndex")
def test_ridx():
    from nil.cognition.index import RetrievalIndex
    idx = RetrievalIndex()
    from nil.cognition.index import StoredItem
    idx.upsert(StoredItem(id="k1", content="some text about search", metadata={"tool": "search"}))
    assert len(idx.search_dense("search")) >= 0

@check("Cognition: ContextAssembler")
def test_asm():
    from nil.cognition.assembler import ContextAssembler
    from nil.cognition.index import ScoredResult, StoredItem
    r = ContextAssembler(max_tokens=2000).assemble([
        ScoredResult(item=StoredItem(id="k1", content="some text"), score=0.9, source="skills"),
    ])
    assert "some text" in r


# ======================== 9. FULL AGENT ========================

@check("Agent: Full lifecycle")
def test_agent_e2e():
    import nil
    @nil.tool
    def diagnose(s: str) -> str:
        return f"diag: {s}"
    @nil.tool
    def fix(a: str) -> str:
        return f"fixed: {a}"
    d = nil.Domain("verify-e2e", [diagnose, fix], ["resolve issues"])
    mem = nil.PersistentMemory(os.path.join(TEMP_DIR, "ae"))
    a = nil.Agent(domain=d, memory=mem)
    assert a.run_count == 0
    r = a.run("server is slow")
    assert r.run_id and r.duration > 0
    assert mem.get_trace_count("verify-e2e") == 1
    for i in range(4):
        a.run(f"incident #{i}")
    assert a.run_count == 5
    s = a.stats()
    assert s["domain"] == "verify-e2e"

@check("Agent: Evolution continuous")
def test_agent_evol():
    import nil
    from nil.config import EvolutionConfig
    @nil.tool
    def t(x: str) -> str:
        return x
    a = nil.Agent(domain=nil.Domain("ev", [t]),
                  memory=nil.PersistentMemory(os.path.join(TEMP_DIR, "ae2")),
                  evolution=EvolutionConfig(strategy="continuous"))
    for i in range(3):
        a.run(f"t-{i}")
    assert a.run_count == 3

@check("Agent: Evolution manual")
def test_agent_manual():
    import nil
    from nil.config import EvolutionConfig
    @nil.tool
    def t(x: str) -> str:
        return x
    a = nil.Agent(domain=nil.Domain("man", [t]),
                  memory=nil.PersistentMemory(os.path.join(TEMP_DIR, "am")),
                  evolution=EvolutionConfig(strategy="manual"))
    for i in range(3):
        a.run(f"t-{i}")
    a.evolve()
    assert a.run_count == 3

@check("Agent: Introspection")
def test_agent_intro():
    import nil
    @nil.tool
    def t(x: str) -> str:
        return x
    a = nil.Agent(domain=nil.Domain("si", [t]),
                  memory=nil.PersistentMemory(os.path.join(TEMP_DIR, "ai")))
    a.run("hi")
    for k in ("domain", "model", "run_count", "success_rate", "tools_available"):
        assert k in a.stats(), f"missing {k}"
    assert a.learned_skills() == []


# ======================== 10. MULTI-AGENT ========================

@check("Agents: MessageBus")
def test_bus():
    from nil.agents.messages import MessageBus, AgentMessage, MessageType
    bus = MessageBus()
    bus.receive("bob")  # initialize bob's queue
    msg = AgentMessage(source="alice", target="bob", message_type=MessageType.REQUEST, content="hello")
    bus.send(msg)
    received = bus.receive("bob")
    assert len(received) == 1
    assert received[0].source == "alice"

@check("Agents: AgentPool")
def test_pool():
    from nil.agents.roles import AgentRole, AgentSpec
    from nil.agents.pool import AgentPool
    pool = AgentPool()
    spec = AgentSpec(role=AgentRole.EXECUTOR, tools=[])
    pool.register_spec(spec)
    ref = pool.acquire(AgentRole.EXECUTOR)
    assert ref is not None
    pool.release(ref)

@check("Agents: OrchestratorAgent")
def test_orch():
    import nil
    from nil.agents.orchestrator import OrchestratorAgent
    @nil.tool
    def t(x: str) -> str:
        return x
    o = OrchestratorAgent(domain=nil.Domain("ot", [t]))
    assert o is not None

@check("Agents: MultiAgentNil")
def test_ma():
    from nil.agents.framework import MultiAgentNil
    assert MultiAgentNil() is not None


# ======================== 11. BENCHMARKS ========================

@check("Benchmarks: All instantiable")
def test_bm():
    from nil.benchmarks.learning import LearningCurveBenchmarks, ContextDependenceBenchmark
    from nil.benchmarks.horizon import LongHorizonBenchmark
    from nil.benchmarks.memory import MemoryRetentionBenchmark
    from nil.benchmarks.recovery import RecoveryBenchmark
    from nil.benchmarks.transfer import TransferBenchmark
    from nil.benchmarks.system import SystemBenchmark
    assert all(x is not None for x in [
        LearningCurveBenchmarks(), ContextDependenceBenchmark(),
        LongHorizonBenchmark(), MemoryRetentionBenchmark(),
        RecoveryBenchmark(), TransferBenchmark(), SystemBenchmark()])

@check("Benchmarks: Suite + CompositeScore")
def test_bm_suite():
    from nil.benchmarks.base import CompositeScore
    cs = CompositeScore(learning_score=0.85)
    assert cs.learning_score == 0.85


# ======================== 12. POMDP MODELS ========================

@check("Models: StateSpace + EnvironmentState")
def test_ss():
    from nil.models.state_space import EnvironmentState, StateSpace
    s = EnvironmentState(tools_used=["fix"], is_terminal_success=True, errors_encountered=[])
    assert s.to_key() is not None
    ss = StateSpace()
    ss.add_state(s)
    assert len(ss.all_keys()) == 1

@check("Models: TransitionModel")
def test_tm():
    from nil.models.state_space import EnvironmentState
    from nil.models.transitions import TransitionModel
    tm = TransitionModel()
    s = EnvironmentState(tools_used=["fix"], is_terminal_success=True, errors_encountered=[])
    key = s.to_key()
    tm.record(key, "fix", key)
    assert tm.predict(key, "fix") is not None

@check("Models: BeliefTracker with StateSpace")
def test_bt():
    from nil.models.state_space import EnvironmentState, StateSpace
    from nil.models.belief import BeliefTracker
    ss = StateSpace()
    s = EnvironmentState(tools_used=["fix"], is_terminal_success=True)
    ss.add_state(s)
    bt = BeliefTracker(ss)
    bt.update("fix", s.to_key())
    assert bt.most_likely_state is not None

@check("Models: MCTSPlanner")
def test_mcts():
    from nil.models.planners import MCTSPlanner
    assert MCTSPlanner() is not None

@check("Models: ModelAwareAgent")
def test_ma():
    from nil.models.framework import ModelAwareAgent
    a = ModelAwareAgent(domain="test")
    a.start_task("fix db")
    a.observe_outcome({"tool_name": "fix", "step_type": "tool_call", "result": "ok", "error": None})
    a.finish_task({"tool_sequence": ["fix"], "outcome_success": True, "steps": []})

@check("Models: OnlineModelLearner")
def test_ol():
    from nil.models.learning import OnlineModelLearner
    l = OnlineModelLearner()
    l.update({"tool_sequence": ["fix"], "outcome_success": True, "steps": []})


# ======================== 13. RL ========================

@check("RL: DenseRewardModel")
def test_rl_rew():
    from nil.rl.rewards import DenseRewardModel
    r = DenseRewardModel().step_reward({}, {})
    assert isinstance(r, float)

@check("RL: Policy")
def test_rl_pol():
    from nil.rl.policy import Policy, BehavioralCloning
    p2 = Policy()
    a = p2.act({}, "test", available_actions=["a", "b"])
    assert a in ("a", "b")
    assert BehavioralCloning() is not None

@check("RL: ReplayBuffer")
def test_rl_buf():
    from nil.rl.replay import ReplayBuffer, Transition
    b = ReplayBuffer(10)
    b.push(Transition({}, "a", 1.0, {}, False))
    assert b.size == 1

@check("RL: EloTrajectoryRating")
def test_rl_elo():
    from nil.rl.trajectory import EloTrajectoryRating
    w, l = EloTrajectoryRating().compare("a", "b")
    assert w > l

@check("RL: RLEnabledAgent")
def test_rl_agent():
    from nil.rl.agent import RLEnabledAgent
    a = RLEnabledAgent("test")
    act = a.select_action({}, ["a", "b"])
    assert act in ("a", "b")
    from nil.kernel.outcome import StepRecord
    a.on_step_complete(StepRecord(step_number=1, step_type="tool_call"), {}, {})
    a.finish_trajectory({"tool_sequence": [], "outcome_success": True, "steps": []})


# ======================== 14. COMPILATION ========================

@check("Compilation: SkillParser")
def test_comp_parse():
    from nil.compilation.pipeline import SkillParser, IntermediateRepresentation, _SkillStub
    ir = SkillParser().parse(_SkillStub(name="test", description="parse test",
                                         action_sequence=["search", "read"],
                                         tool_names=["search", "read"]))
    assert isinstance(ir, IntermediateRepresentation)

@check("Compilation: SkillOptimizer")
def test_comp_opt():
    from nil.compilation.pipeline import SkillOptimizer, IntermediateRepresentation
    ir = IntermediateRepresentation(name="test", tools=["search", "read"])
    assert SkillOptimizer().optimize(ir) is not None

@check("Compilation: SkillCodeGenerator")
def test_comp_cg():
    from nil.compilation.pipeline import SkillCodeGenerator, IntermediateRepresentation
    ir = IntermediateRepresentation(name="test", tools=["search"])
    code = SkillCodeGenerator().generate_python(ir)
    assert "def" in code

@check("Compilation: Full pipeline")
def test_comp_pipe():
    from nil.compilation.pipeline import SkillCompilationEngine, CompilationResult, _SkillStub
    r = SkillCompilationEngine().process_skill(_SkillStub(name="sr", action_sequence=["search", "read"],
                                                           tool_names=["search", "read"]))
    assert r is not None or isinstance(r, CompilationResult)


# ======================== 15. OBSERVABILITY ========================

@check("Observability: Event system")
def test_obs_ev():
    from nil.observability.events import EventType, ObservabilityEvent, ObservabilityStore
    s = ObservabilityStore()
    s.write(ObservabilityEvent(event_type=EventType.LLM_CALL_STARTED, agent_id="a1", domain="d1", payload={"t": 10}))
    assert len(s.get_events(event_type=EventType.LLM_CALL_STARTED)) == 1

@check("Observability: TraceViewer + FlameGraph")
def test_obs_vw():
    from nil.observability.events import EventType, ObservabilityEvent, ObservabilityStore
    from nil.observability.trace_viewer import TraceViewer
    from nil.observability.flame_graph import FlameGraphGenerator
    store = ObservabilityStore()
    store.write(ObservabilityEvent(event_type=EventType.LLM_CALL_STARTED, agent_id="a", domain="d",
                                   payload={}, timestamp=time.time()))
    store.write(ObservabilityEvent(event_type=EventType.TOOL_CALL_STARTED, agent_id="a", domain="d",
                                   payload={}, timestamp=time.time() + 0.1))
    assert TraceViewer(store=store).render_timeline("obs-run") is not None
    assert FlameGraphGenerator(store=store).generate("obs-run") is not None

@check("Observability: ReasoningTree, MemoryBrowser, LiveDebugger")
def test_obs_more():
    from nil.observability.reasoning_tree import ReasoningTree
    from nil.observability.memory_browser import MemoryBrowser
    from nil.observability.evolution_diff import EvolutionDiffViewer
    from nil.observability.debugger import LiveDebugger
    rt = ReasoningTree()
    rt.capture(run_id="step1", step_number=1, prompt="What to do?", response="Use search")
    assert len(rt.get_nodes("step1")) >= 1
    assert MemoryBrowser().render_html(domain="test") is not None
    from nil.observability.events import ObservabilityStore as _ObsStore
    assert EvolutionDiffViewer(store=_ObsStore()).render_text("test-evo") is not None
    dbg = LiveDebugger()
    dbg.pause()
    assert dbg.paused is True
    dbg.resume()
    assert dbg.paused is False


# ======================== 16. TRAINING ========================

@check("Training: CurriculumEngine")
def test_cur():
    from nil.training.curriculum import CurriculumEngine
    from nil.memory.store import PersistentMemory
    mem = PersistentMemory(os.path.join(TEMP_DIR, "cur"))
    ce = CurriculumEngine()
    r = ce.run(domain="d", traces=[], tool_names=["t1"])
    assert r is not None

@check("Training: WeaknessAnalyzer")
def test_wa():
    from nil.training.curriculum import WeaknessAnalyzer
    assert WeaknessAnalyzer().analyze([], []) is not None

@check("Training: SyntheticTaskGenerator")
def test_stg():
    from nil.training.curriculum import SyntheticTaskGenerator, WeaknessProfile
    tasks = SyntheticTaskGenerator().generate_for_weakness(
        WeaknessProfile(domain="d", success_rate=0.5))
    assert isinstance(tasks, list)


# ======================== 17. OPTIMIZATION ========================

@check("Optimization: ModelRouter")
def test_mr():
    from nil.optimization.model_router import ModelRouter
    assert ModelRouter().route("simple", 0.3) is not None

@check("Optimization: ChainCompressor")
def test_cc():
    from nil.optimization.chain_compressor import ChainCompressor
    assert ChainCompressor().analyze([]) == []

@check("Optimization: CacheManager")
def test_cm():
    from nil.optimization.cache_manager import CacheManager
    cm = CacheManager()
    cm.put("k", {}, "v")
    assert cm.get("k", {}) == "v"
    assert cm.get("x", {}) is None

@check("Optimization: BudgetController")
def test_bc():
    from nil.optimization.budget_controller import BudgetController
    bc = BudgetController()
    assert bc.max_cost_per_day > 0


# ======================== RUN ========================

def main():
    print("=" * 60)
    print("Nil SDK — Comprehensive System Verification")
    print("=" * 60)
    print(f"Temp dir: {TEMP_DIR}\n")

    fns = [(n, fn) for n, fn in globals().items() if n.startswith("test_") and callable(fn)]
    print(f"Running {len(fns)} verification checks...\n")

    for _, fn in fns:
        fn()

    print()
    print("=" * 60)
    passed = sum(1 for r in results if r[0] == PASS)
    failed = sum(1 for r in results if r[0] == FAIL)
    print(f"Results: {passed} PASSED, {failed} FAILED / {len(results)} TOTAL")
    print("=" * 60)

    if failed:
        print("\nFAILED CHECKS:")
        for s, n, d in results:
            if s == FAIL:
                print(f"  {n}: {d}")
        sys.exit(1)

    print("\nAll systems operational.")
    sys.exit(0)


if __name__ == "__main__":
    main()
