"""User-facing smoke tests for nil-sdk — exercises the full public API."""

import os
import shutil

import nil


@nil.tool
def search_logs(query: str) -> str:
    """Search application logs for errors or patterns."""
    return f"Found entries matching '{query}'"


@nil.tool
def apply_fix(action: str) -> str:
    """Apply a remediation action."""
    return f"Applied fix: {action}"


@nil.tool
def verify(check: str) -> str:
    """Verify a fix was successful."""
    return f"Verification passed: {check}"


STORE = "./_test_store"


def setup():
    if os.path.exists(STORE):
        shutil.rmtree(STORE)


def test_tool_decorator_generates_schema():
    assert hasattr(search_logs, "name")
    assert search_logs.name == "search_logs"
    schema = search_logs.to_schema()
    assert "query" in schema["function"]["parameters"]["properties"]


def test_domain_creation():
    domain = nil.Domain(
        name="test-domain",
        tools=[search_logs, apply_fix],
        objectives=["Test the system"],
    )
    assert domain.name == "test-domain"
    assert "search_logs" in domain.tool_name


def test_memory_store_and_retrieve():
    memory = nil.PersistentMemory(STORE)
    memory.store_trace({
            "run_id": "test-1",
            "domain": "test",
            "task": "test task",
            "steps": [],
            "outcome_success": True,
            "outcome_output": "done",
            "total_duration": 1.0,
            "total_tokens_in": 10,
            "total_tokens_out": 5,
            "total_cost": 0.001,
            "tool_sequence": ["search_logs"],
            "error": None,
            "created_at": 1000,
            "metadata": {},
        })
    traces = memory.get_traces("test")
    assert len(traces) >= 1


def test_agent_run_with_stub():
    domain = nil.Domain(name="test-stub", tools=[search_logs, apply_fix])
    agent = nil.Agent(domain=domain)
    result = agent.run("test task")
    assert hasattr(result, "success")
    assert hasattr(result, "steps")
    assert hasattr(result, "duration")
    assert result.success is True
    assert len(result.steps) > 0


def test_agent_run_returns_tool_sequence():
    domain = nil.Domain(name="test-tools", tools=[search_logs, apply_fix, verify])
    agent = nil.Agent(domain=domain)
    result = agent.run("find and fix the bug")
    assert isinstance(result.tool_sequence, list)
    assert len(result.tool_sequence) > 0


def test_agent_introspection():
    domain = nil.Domain(name="test-intro", tools=[search_logs])
    agent = nil.Agent(domain=domain, memory=nil.PersistentMemory(STORE))
    agent.run("first task")
    agent.run("second task")
    assert agent.run_count >= 2
    assert isinstance(agent.success_rate(), float)
    assert isinstance(agent.learned_skills(), list)
    assert isinstance(agent.stats(), dict)
    assert "run_count" in agent.stats()


def test_agent_evolution_config():
    domain = nil.Domain(name="test-evo", tools=[search_logs, apply_fix])
    agent = nil.Agent(
        domain=domain,
        evolution=nil.EvolutionConfig(
            enabled=True,
            strategy="periodic",
            evolve_interval=2,
        ),
    )
    agent.run("task 1")
    agent.run("task 2")
    assert agent.run_count == 2
    # evolution may or may not have triggered depending on trace count


def test_agent_with_success_signal():
    def always_fail(outcome):
        return False

    domain = nil.Domain(
        name="test-signal",
        tools=[search_logs],
        success_signal=always_fail,
    )
    agent = nil.Agent(domain=domain)
    result = agent.run("any task")
    assert result.success is False


def test_multiple_runs_increment_count():
    domain = nil.Domain(name="test-count", tools=[search_logs])
    agent = nil.Agent(domain=domain, memory=nil.PersistentMemory(STORE))
    for i in range(5):
        agent.run(f"task {i}")
    assert agent.run_count == 5


def test_e2e_full_workflow():
    domain = nil.Domain(
        name="test-e2e",
        tools=[search_logs, apply_fix, verify],
        objectives=["Resolve issues"],
    )
    agent = nil.Agent(
        domain=domain,
        evolution=nil.EvolutionConfig(enabled=True, strategy="continuous"),
    )

    tasks = [
        "API timeout on /users endpoint",
        "Database connection pool exhausted",
        "Memory leak in worker process",
    ]

    for task in tasks:
        result = agent.run(task)
        assert hasattr(result, "success")
        assert hasattr(result, "tool_sequence")

    stats = agent.stats()
    assert stats["run_count"] == len(tasks)
    assert stats["domain"] == "test-e2e"
    assert stats["tools_available"] >= 3


def test_domain_name_normalization():
    d = nil.Domain(name="  My Domain  ", tools=[])
    assert d.name == "my-domain"


def test_memory_success_rate():
    memory = nil.PersistentMemory(STORE)
    for i in range(10):
        memory.store_trace({
            "run_id": f"rate-{i}",
            "domain": "rate-test",
            "task": "task",
            "steps": [],
            "outcome_success": i < 7,
            "outcome_output": "",
            "total_duration": 1.0,
            "total_tokens_in": 10,
            "total_tokens_out": 5,
            "total_cost": 0.001,
            "tool_sequence": [],
            "error": None,
            "created_at": 1000 + i,
            "metadata": {},
        })
    rate = memory.get_success_rate("rate-test", last_n=10)
    assert rate == 0.7
