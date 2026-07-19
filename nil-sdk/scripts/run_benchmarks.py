"""Run the full NilBenchmarkSuite against a nil Agent.

Usage:
    python run_benchmarks.py                # full benchmark (slow)
    python run_benchmarks.py --quick         # fast smoke-test (10 system runs)
    python run_benchmarks.py --smoke         # minimal: 1 run per benchmark
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import nil
from nil.benchmarks.base import Outcome as BenchmarkOutcome
from nil.benchmarks.runner import NilBenchmarkSuite
from nil.kernel.outcome import RunOutcome


@nil.tool
def search_logs(query: str) -> str:
    """Search application logs for errors or patterns."""
    responses = {
        "timeout": "Found 3 entries: [ERROR] Connection timeout at 14:32, 14:35, 14:41",
        "memory": "Found 2 entries: [WARN] Heap usage 94% at 13:00, [ERROR] OOM killer at 13:05",
        "ssl": "Found 1 entry: [WARN] Certificate expires in 24h",
        "disk": "Found 4 entries: [WARN] /var/log 92% full, [ERROR] Write failed",
        "database": "Found 2 entries: [ERROR] Connection pool exhausted, [WARN] Slow query 4.2s",
    }
    for key, response in responses.items():
        if key in query.lower():
            return response
    return f"No critical logs found for: {query}"


@nil.tool
def check_status(service: str) -> str:
    """Check if a service is running and healthy."""
    return f"Service '{service}': running, response time 2.3s (degraded)"


@nil.tool
def apply_fix(action: str) -> str:
    """Apply a remediation action."""
    return f"Fix applied: {action}. Service recovering."


@nil.tool
def verify(check: str) -> str:
    """Verify a fix was successful."""
    return f"Verification passed: {check} — all green."


class BenchmarkAgent:
    """Wraps a nil.Agent so RunOutcome conforms to the benchmark Outcome protocol."""

    def __init__(self, agent: nil.Agent):
        self._agent = agent

    def __repr__(self) -> str:
        return repr(self._agent)

    def run(self, task: str) -> BenchmarkOutcome:
        ro: RunOutcome = self._agent.run(task)
        return BenchmarkOutcome(
            success=ro.success,
            steps=[
                {
                    "step_number": s.step_number,
                    "step_type": s.step_type,
                    "tool_name": s.tool_name or "",
                    "tool_input": s.tool_input or {},
                    "tool_output": s.tool_output,
                    "duration_ms": s.duration_ms,
                    "tokens_in": s.tokens_in,
                    "tokens_out": s.tokens_out,
                    "error": s.error,
                    "success": s.error is None,
                    "timestamp": s.timestamp,
                }
                for s in ro.steps
            ],
            total_cost=ro.total_cost,
            duration=ro.duration,
            error=ro.error or "",
            evolution_duration=ro.evolution_duration,
            evolution_triggered=ro.evolution_triggered,
            had_errors=ro.had_errors,
            error_steps=[
                {
                    "step_number": s.step_number,
                    "step_type": s.step_type,
                    "tool_name": s.tool_name or "",
                    "error": s.error or "",
                }
                for s in ro.steps
                if s.error
            ],
        )


STORE_DIR = "./benchmark_store"
RESULTS_DIR = "./benchmark_results"


def _provider_name() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    return "stub"


def main():
    parser = argparse.ArgumentParser(description="Nil SDK Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Fast smoke-test (reduced runs)")
    parser.add_argument("--smoke", action="store_true", help="Minimal: 1 run per benchmark")
    args = parser.parse_args()

    provider = _provider_name()
    print(f"Provider: {provider}")
    if provider == "stub":
        print("No API key set — using StubProvider (offline, deterministic).")
        print("Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY for real LLM benchmarks.\n")

    for d in [STORE_DIR, RESULTS_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)

    domain = nil.Domain(
        name="incident-response",
        tools=[search_logs, check_status, apply_fix, verify],
        objectives=[
            "Diagnose and resolve production incidents",
            "Always search logs before applying fixes",
            "Verify fixes after applying them",
        ],
    )

    agent = nil.Agent(
        domain=domain,
        base_model="llama3.2",
        memory=nil.PersistentMemory(STORE_DIR),
        provider="openai",
        evolution=nil.EvolutionConfig(
            strategy="periodic",
            evolve_interval=10,
            skill_extraction=True,
            drift_detection=True,
        ),
    )

    bench_agent = BenchmarkAgent(agent)

    suite = NilBenchmarkSuite()

    if args.smoke:
        suite.system.n_runs = 1
    elif args.quick:
        suite.system.n_runs = 10

    report = suite.run(bench_agent, output_dir=RESULTS_DIR)

    print()
    print(report.summary())
    print()
    print(f"Full JSON report: {RESULTS_DIR}/benchmark_report.json")

    exit_code = 0 if report.composite.total >= 0.4 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
