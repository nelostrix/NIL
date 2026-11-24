# Workflow coverage

## Usage

```python
import nil
print(nil.coverage.describe('native.tensor'))
print(nil.coverage.list_workflows(area=15))
print(nil.coverage.areas())
print(nil.coverage.availability('neural.supervised'))
```

## Parameters

`nil workflows` lists the bounded inventory. Add a workflow ID and `--json` for
its API, input/output descriptions, limits, backend requirements and evidence.
`--availability` inspects installed distribution metadata without importing a
numerical backend, starting workers, probing devices or contacting services.

## Limitations

Implementation kinds: native (NIL-owned), integration (external numerical
backend), hybrid (both), unimplemented. Validation is separate: validated means
recorded evidence for the stated subset, target_blocked means acceptance is
outstanding. Neither indicates current-machine readiness. Installed versions may
not match recorded profiles. All 32 roadmap areas remain partial or not_audited;
this inventory is not a claim of complete framework coverage.

## Output

`matrix()` returns independent versioned metadata. Unknown IDs and filters raise
ValueError. Schema 1 is an alpha contract; SDK and matrix versions are separate.

## Requirements

Python standard library only; distribution metadata is read without importing backends.
