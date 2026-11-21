# lifecycle

Store local run metadata, metrics, events and versioned checkpoint artifacts.

## Usage

```python
import nil

run = nil.Run("./nil_runs", name="example")
run.status("running")
run.log({"loss": 0.25}, step=1)
run.status("completed")
print(run.path)
```

## Parameters

Run root/name/config; log takes finite metrics and a step. Registry needs a local root.

## Output

Local run files and explicit versioned artifact metadata.

## Requirements

Core SDK for records; native model checkpoint loading needs its backend.

## Limitations

Writes local files. SHA-256 detects corruption, not authenticity. It is not a remote model registry.
