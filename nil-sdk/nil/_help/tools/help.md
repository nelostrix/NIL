# tools

Define typed callables for the optional agent tool runtime.

## Usage

```python
import nil

@nil.tool
def square(value: float) -> float:
    """Square a number."""
    return value * value

print(square.name)
```

## Parameters

Typed function, docstring and supported structured arguments.

## Output

Tool metadata/schema and an executable callable managed by the agent kernel.

## Requirements

Core SDK.

## Limitations

Tools execute trusted Python with its process privileges. Tool definitions do not provide an OS sandbox.
