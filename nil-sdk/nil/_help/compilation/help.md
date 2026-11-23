# compilation

Optional conversion of learned tool workflows into explicit executable plans.

## Usage

```python
from nil.compilation import SkillCompilationEngine

engine = SkillCompilationEngine()
print(type(engine).__name__)
```

## Parameters

Skill descriptions/action sequences and explicit tool registrations.

## Output

Compilation plans/results and workflow execution objects.

## Requirements

Core SDK.

## Limitations

This is agent workflow compilation, not a replacement for a neural compiler or a sandbox for untrusted code.
