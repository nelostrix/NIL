# kernel

Optional agent tool execution, tracing and graph scheduling.

## Usage

```python
from nil.kernel.tracer import Tracer

tracer = Tracer(run_id="example", domain="demo")
print(type(tracer).__name__)
```

## Parameters

Explicit run/domain, registered tools and provider configuration.

## Output

Execution traces and agent outcomes.

## Requirements

Core SDK; real model providers use optional dependencies.

## Limitations

The kernel is separate from tensor execution and should be invoked only for agent workflows.
