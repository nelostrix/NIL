# run_real.py — your first real evolution loop

import os

os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"

import nil


# Define tools that simulate a real debugging workflow
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


# Create the domain
domain = nil.Domain(
    name="incident-response",
    tools=[search_logs, check_status, apply_fix, verify],
    objectives=[
        "Diagnose and resolve production incidents",
        "Always search logs before applying fixes",
        "Verify fixes after applying them",
    ],
)

# Create the agent with evolution enabled
agent = nil.Agent(
    domain=domain,
    base_model="llama3.2",
    memory=nil.PersistentMemory("./real_store"),
    provider="openai",
    evolution=nil.EvolutionConfig(
        strategy="periodic",
        evolve_interval=5,  # Evolve every 5 runs
        skill_extraction=True,
        drift_detection=True,
    ),
)

# Run 10 tasks — evolution triggers at run 5 and run 10
tasks = [
    "API timeout on /v2/users endpoint",
    "Database connection pool exhausted",
    "Memory leak detected in worker process",
    "SSL certificate expiring in 24 hours",
    "Disk full on logging server",
    "API latency spike on /v2/orders",
    "Database slow queries detected",
    "Memory usage at 95% on production",
    "SSL handshake failures on port 443",
    "Disk I/O bottleneck on database server",
]

print("=" * 60)
print("NIL — Self-Evolving Agent Demo")
print("=" * 60)

for i, task in enumerate(tasks):
    print(f"\n--- Run {i + 1}: {task} ---")
    result = agent.run(task)
    print(f"  Success: {result.success}")
    print(f"  Tools used: {result.tool_sequence}")
    print(f"  Duration: {result.duration:.1f}s")
    if result.output:
        print(f"  Output: {str(result.output)[:150]}")
    if result.error:
        print(f"  Error: {result.error}")

# Show what the agent learned
print("\n" + "=" * 60)
print("WHAT THE AGENT LEARNED")
print("=" * 60)
print(f"\nTotal runs: {agent.run_count}")
print(f"Success rate: {agent.success_rate():.0%}")
print(f"Skills learned: {agent.learned_skills()}")
print(f"Evolution history: {len(agent.evolution_history())} cycles")
print(f"Current strategy: {agent.get_strategy()}")

for i, evo in enumerate(agent.evolution_history()):
    print(f"\nEvolution #{i + 1}:")
    print(f"  Stages: {evo['stages']}")
    print(f"  Skills extracted: {evo['skills_extracted']}")
    print(f"  Deltas applied: {evo['deltas_applied']}")
    print(f"  Drift detected: {evo['drift_detected']}")
    print(f"  Duration: {evo['duration_s']}s")
