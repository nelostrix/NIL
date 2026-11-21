# models

Legacy environment/POMDP modeling, separate from the trainable-model catalog.

## Usage

```python
from nil.models import StateSpace

space = StateSpace()
print(type(space).__name__)
```

## Parameters

Environment states, transition/observation/reward models and planning configuration.

## Output

Environment/planning objects.

## Requirements

Core SDK.

## Limitations

Use nil.model(name) for trainable models. Legacy environment models are optional and are not neural compute backends.
