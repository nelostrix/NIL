# agent

Use the existing optional tool-using agent application with explicit domain and memory.

## Usage

```python
import nil

@nil.tool
def greet(name: str) -> str:
    """Return a greeting."""
    return f"Hello {name}"

domain = nil.Domain(name="demo", tools=[greet], objectives=["greet a user"])
agent = nil.Agent(domain, memory=nil.PersistentMemory("./agent_store"), provider="stub", evolution=None)
result = agent.run("greet Ada")
print(result.output)
```

## Parameters

Domain/tools, memory, provider/model, explicit evolution options.

## Output

A RunOutcome and local agent records.

## Requirements

Core SDK with the explicit stub test provider; real providers require their optional extras and configuration.

## Limitations

The stub is an offline test fixture, not real intelligence. Real providers may incur cost. The agent application is separate from model training.
