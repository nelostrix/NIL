# memory

Optional local agent experience and skill storage.

## Usage

```python
import nil

memory = nil.PersistentMemory("./agent_store")
print(type(memory).__name__)
```

## Parameters

Memory directory and explicit queries/records for each memory API.

## Output

SQLite-backed records and query results.

## Requirements

Core SDK.

## Limitations

Creates local persistent state. Agent experience storage is not a neural tensor dataset or model checkpoint.
