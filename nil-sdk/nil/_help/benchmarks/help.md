# benchmarks

Optional agent behavior and synthetic benchmark utilities.

## Usage

```python
from nil.benchmarks import NilBenchmarkSuite

suite = NilBenchmarkSuite()
print(type(suite).__name__)
```

## Parameters

Explicit benchmark task selection, agent configuration and seeds.

## Output

Benchmark objects and reports after explicit runs.

## Requirements

Core SDK; executed agents may require providers.

## Limitations

Synthetic tests are not evidence of production task quality, accelerator speed or universal AI coverage.
