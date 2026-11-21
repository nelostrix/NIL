# agents

Optional multi-agent messaging and coordination helpers.

## Usage

```python
from nil.agents import MessageBus

bus = MessageBus(agent_name="worker")
print(type(bus).__name__)
```

## Parameters

Agent names, task/domain configuration, tools, and optional callbacks according to the selected coordinator.

## Output

Coordination/message objects; task execution depends on configured agents.

## Requirements

Core SDK; actual model providers may need optional extras.

## Limitations

Constructing a coordinator does not prove task quality or distributed infrastructure. Ordinary model operations do not load this subsystem.
