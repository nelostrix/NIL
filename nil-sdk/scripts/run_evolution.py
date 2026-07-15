"""Test that actually triggers evolution by simulating mixed outcomes."""

import os
import time

os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"

import nil
from nil.evolution.pipeline import EvolutionPipeline

# Create memory fresh
memory = nil.PersistentMemory("./evolution_test_store")

# Manually insert realistic traces with VARIETY
# Some succeed with good patterns, some fail with bad patterns
traces = []
for i in range(8):
    traces.append(
        {
            "run_id": f"good-{i}",
            "domain": "ops",
            "task": f"Fix server timeout issue #{i}",
            "steps": [
                {
                    "step_number": 1,
                    "step_type": "tool_call",
                    "tool_name": "search_logs",
                    "tool_input": {"query": "timeout"},
                    "tool_output": "Found errors",
                    "duration_ms": 50,
                    "tokens_in": 10,
                    "tokens_out": 5,
                    "error": None,
                    "timestamp": time.time(),
                },
                {
                    "step_number": 2,
                    "step_type": "tool_call",
                    "tool_name": "diagnose",
                    "tool_input": {"symptom": "timeout"},
                    "tool_output": "Root cause found",
                    "duration_ms": 50,
                    "tokens_in": 10,
                    "tokens_out": 5,
                    "error": None,
                    "timestamp": time.time(),
                },
                {
                    "step_number": 3,
                    "step_type": "tool_call",
                    "tool_name": "apply_fix",
                    "tool_input": {"action": "restart"},
                    "tool_output": "Fixed",
                    "duration_ms": 50,
                    "tokens_in": 10,
                    "tokens_out": 5,
                    "error": None,
                    "timestamp": time.time(),
                },
            ],
            "outcome_success": True,
            "outcome_output": "Resolved",
            "total_duration": 2.0,
            "total_tokens_in": 100,
            "total_tokens_out": 50,
            "total_cost": 0.003,
            "tool_sequence": ["search_logs", "diagnose", "apply_fix"],
            "error": None,
            "created_at": time.time() + i,
            "metadata": {},
        }
    )

for i in range(5):
    traces.append(
        {
            "run_id": f"bad-{i}",
            "domain": "ops",
            "task": f"Handle database error #{i}",
            "steps": [
                {
                    "step_number": 1,
                    "step_type": "tool_call",
                    "tool_name": "apply_fix",
                    "tool_input": {"action": "restart db"},
                    "tool_output": "Failed",
                    "duration_ms": 50,
                    "tokens_in": 10,
                    "tokens_out": 5,
                    "error": "ConnectionRefused",
                    "timestamp": time.time(),
                },
            ],
            "outcome_success": False,
            "outcome_output": "Failed",
            "total_duration": 1.0,
            "total_tokens_in": 50,
            "total_tokens_out": 20,
            "total_cost": 0.001,
            "tool_sequence": ["apply_fix"],
            "error": "Tried to fix without diagnosing first",
            "created_at": time.time() + 10 + i,
            "metadata": {},
        }
    )

for i in range(4):
    traces.append(
        {
            "run_id": f"verify-{i}",
            "domain": "ops",
            "task": f"Fix memory leak #{i}",
            "steps": [
                {
                    "step_number": 1,
                    "step_type": "tool_call",
                    "tool_name": "search_logs",
                    "tool_input": {"query": "memory"},
                    "tool_output": "OOM found",
                    "duration_ms": 50,
                    "tokens_in": 10,
                    "tokens_out": 5,
                    "error": None,
                    "timestamp": time.time(),
                },
                {
                    "step_number": 2,
                    "step_type": "tool_call",
                    "tool_name": "apply_fix",
                    "tool_input": {"action": "restart worker"},
                    "tool_output": "Restarted",
                    "duration_ms": 50,
                    "tokens_in": 10,
                    "tokens_out": 5,
                    "error": None,
                    "timestamp": time.time(),
                },
                {
                    "step_number": 3,
                    "step_type": "tool_call",
                    "tool_name": "verify",
                    "tool_input": {"check": "memory"},
                    "tool_output": "All clear",
                    "duration_ms": 50,
                    "tokens_in": 10,
                    "tokens_out": 5,
                    "error": None,
                    "timestamp": time.time(),
                },
            ],
            "outcome_success": True,
            "outcome_output": "Fixed and verified",
            "total_duration": 3.0,
            "total_tokens_in": 100,
            "total_tokens_out": 50,
            "total_cost": 0.005,
            "tool_sequence": ["search_logs", "apply_fix", "verify"],
            "error": None,
            "created_at": time.time() + 20 + i,
            "metadata": {},
        }
    )

for t in traces:
    memory.store_trace(t)

print(
    f"Inserted {len(traces)} traces ({sum(1 for t in traces if t['outcome_success'])} success, {sum(1 for t in traces if not t['outcome_success'])} failures)"
)

# Run evolution pipeline
pipeline = EvolutionPipeline(memory, min_traces_for_evolution=10)
result = pipeline.run("ops")

print(f"\n{'=' * 60}")
print("EVOLUTION RESULTS")
print(f"{'=' * 60}")
print(f"Stages run: {result.stages_run}")
print(f"Skills extracted: {result.skills_extracted}")
print(f"Skills deprecated: {result.skills_deprecated}")
print(f"Deltas applied: {result.deltas_applied}")
print(f"Strategy promoted: {result.strategy_promoted}")
print(f"Drift detected: {result.drift_detected}")
print(f"Duration: {result.duration:.2f}s")

if result.analysis:
    print(f"\nAnalysis:")
    print(f"  Total traces: {result.analysis.total_traces}")
    print(f"  Success rate: {result.analysis.baseline_success_rate:.0%}")
    print(f"  Significant patterns: {len(result.analysis.significant_patterns)}")
    print(f"  Systemic failures: {len(result.analysis.systemic_failures)}")
    for p in result.analysis.significant_patterns[:3]:
        print(f"  Pattern: {p.sequence} — {p.success_rate:.0%} success, p={p.p_value:.4f}")

if result.extraction:
    print(f"\nSkills:")
    for s in result.extraction.extracted:
        print(f"  {s.name}: {s.compressed_instruction[:100]}")
        print(f"    Trigger: {s.trigger.description}")
        print(f"    Confidence: {s.confidence:.0%}")

if result.gradient_batch:
    print(
        f"\nGradients: {result.gradient_batch.gradient_count} critiques → {len(result.gradient_batch.deltas)} deltas"
    )
    for d in result.gradient_batch.deltas[:5]:
        print(f"  Delta: {d.mutation_type} — {d.description[:80]}")

strategy = memory.get_active_strategy("ops")
if strategy:
    print(f"\nActive strategy:")
    for key, val in strategy.items():
        print(f"  {key}: {val}")

if result.world_model:
    print(f"\nWorld model:")
    print(f"  Tools: {[t.name for t in result.world_model.tool_profiles]}")
    print(f"  Constraints: {len(result.world_model.constraints)}")
    for c in result.world_model.constraints[:5]:
        print(f"    {c.description}")
    print(
        f"  Health: {result.world_model.health.success_rate:.0%} success, trend={result.world_model.health.trend}"
    )

skills = memory.list_skills("ops")
if skills:
    print(f"\nStored skills: {skills}")
