# documentation

Read a public module or model help.md offline.

## Usage

```python
import nil
from nil.documentation import topics

print(topics())
print(nil.help("linear_regression"))
```

## Parameters

Topic identifier; nil help --list lists all bundled topics.

## Output

Markdown text or a sorted topic list.

## Requirements

Core SDK.

## Limitations

Help describes supported interfaces, not tested hardware readiness. Custom models can register an explicit help_path.
