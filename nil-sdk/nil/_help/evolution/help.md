# evolution

Analyze recorded agent traces and explicitly evaluate strategy/skill changes.

## Usage

```python
from nil.evolution.analyzer import TraceAnalyzer

analyzer = TraceAnalyzer()
print(type(analyzer).__name__)
```

## Parameters

Recorded traces, thresholds, memory and explicitly configured evaluation/provider policies.

## Output

Analysis reports or evolution pipeline results.

## Requirements

Core SDK; real LLM critique requires a provider extra.

## Limitations

Evolution is an agent application, not gradient-based neural training. Improvements require measured evaluation and are not guaranteed.
