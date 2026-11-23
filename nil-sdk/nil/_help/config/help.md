# config

Configuration records for the optional agent application.

## Usage

```python
from nil.config import EvolutionConfig, NilConfig

config = NilConfig(timeout=30.0)
evolution = EvolutionConfig(enabled=False)
print(config.timeout, evolution.enabled)
```

## Parameters

Timeout/retry/log settings and explicit evolution policies.

## Output

Configuration dataclasses.

## Requirements

Core SDK.

## Limitations

Configuration alone does not implement an unavailable backend or guarantee autonomous improvements.
