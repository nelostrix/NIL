# domain

Define the tools, objectives and outcome criterion for an agent application.

## Usage

```python
import nil

domain = nil.Domain(name="demo", tools=[], objectives=["answer a question"])
print(domain.name)
```

## Parameters

Name, tool list, objectives and optional success_signal callback.

## Output

A Domain definition.

## Requirements

Core SDK.

## Limitations

A domain definition does not provide a trained model or actual tool implementation.
