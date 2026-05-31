import nil


@nil.tool
def fake_search(query: str) -> str:
    """simulate searching logs."""
    return f"found 3 results for: {query}"


@nil.tool
def fake_fix(action: str) -> str:
    """simulate applying a fix."""
    return f"Applied: {action}"


domain = nil.Domain(
    name="test-domain",
    tools=[fake_search, fake_fix],
    objectives=["resolve test issues"],
    success_signal=lambda run: len(run.steps) >= 0,
)
agent = nil.Agent(
    domain=domain,
    memory=nil.PersistentMemory("./test_store"),
    evolution=nil.EvolutionConfig(strategy="continuous"),
)


for i in range(12):
    result = agent.run(f"Test task #{i}: investigate high latency")
    print(f"Run {i}: success={result.success}, steps={len(result.steps)}")


# Introspect what i learned
print("skills:", agent.learned_skills())
print("success_rate", agent.success_rate())
print("stats:", agent.stats())
# After creating domain, print this:
print(domain.tools)  # see if they're there
print(agent.stats()["tools_available"])  # should be 2
