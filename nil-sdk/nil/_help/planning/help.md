# planning

## Usage

```python
import nil
print(nil.planning.shortest_path('a', 'c', {'a':['b'], 'b':['c']}))
# nil.planning.run(steps, tools, allowed=names, journal='trace.json', tool_versions=versions)
```

## Parameters

Ordered id/tool/arguments steps, prior-result $ref, allowed tool names, explicit versions, per-call timeout, max_steps, resume and cancellation event.

## Output

Plain JSON journal with committed completed prefix; shortest unweighted string-state path.

## Requirements

Standard library and explicitly supplied existing Tool objects. Calls start owned workers only when run is invoked.

## Limitations

Trusted importable tools; default32 steps,16KiB results,1MiB journal. Interrupted external side effects require reconciliation; no automatic retries or concurrent journal ownership.

See docs/SEVEN_WORKFLOWS.md for v1 acceptance and docs/ADVANCED_WORKFLOWS.md for this phase and remaining work.

## Bounded policy loop

agent(goal, policy_tool, tools, policy_version=..., allowed=..., tool_versions=..., journal_root=...) journals each owned decision/action. Failed calls require reconcile(..., result=..., evidence=...). Running calls cannot be reconciled. Caller ensures exclusive journal ownership; no automatic retry or general reasoning guarantee.
