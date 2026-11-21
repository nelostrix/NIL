# providers

Explicit adapters for agent LLM services and an offline test provider.

## Usage

```python
from nil.config import NilConfig
from nil.providers.stub import StubProvider

provider = StubProvider(model="offline-test", config=NilConfig())
print(type(provider).__name__)
```

## Parameters

Model identifier and NilConfig; real providers need their API configuration.

## Output

Provider adapter objects.

## Requirements

Core SDK for stub; anthropic/openai/gemini extras for the corresponding real providers.

## Limitations

The stub fabricates deterministic test responses and must not be presented as a real model. Real API calls can cost money.
