# observability

Optional agent event recording and trace-inspection utilities.

## Usage

```python
from nil.observability import EventRecorder

recorder = EventRecorder()
print(type(recorder).__name__)
```

## Parameters

Events, span/run identifiers and chosen storage/view configuration.

## Output

Structured events and reports.

## Requirements

Core SDK.

## Limitations

Instrumentation does not improve model quality by itself. Avoid recording secrets in event payloads.
