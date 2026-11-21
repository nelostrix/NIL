# cognition

Optional agent loop and retrieval components.

## Usage

```python
from nil.cognition import CognitiveLoop

loop = CognitiveLoop()
print(type(loop).__name__)
```

## Parameters

Plugins, policy configuration, task/domain and retrieval inputs.

## Output

Loop configuration and, after explicit execution, a LoopResult.

## Requirements

Core SDK; selected provider integrations may require optional extras.

## Limitations

An empty loop has no task capabilities. Model catalog and ordinary neural training do not import this subsystem.
