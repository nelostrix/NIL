# runtime

Discover installed packages and explicitly select a native compute device.

## Usage

```python
import nil

print(nil.runtime.capabilities())
```

## Parameters

`capabilities()` lists packages without importing frameworks; `device(name)` selects a torch device; `seed(value)` seeds Python and torch.

## Output

Capability metadata, a native device, or seed side effects.

## Requirements

Core SDK for capabilities; torch for device/seed.
The base wheel installs only NIL. Missing optional backends raise
`MissingDependencyError`; NIL never installs or downloads them automatically.
Selecting an installation extra explicitly adds its external dependencies.

## Limitations

Installed packages do not establish accelerator readiness. Seeding does not guarantee identical GPU results.
Zero dependencies for installation does not mean the current engine adapters can
train without their engines. Use a built wheel to avoid temporary source-build tools.
