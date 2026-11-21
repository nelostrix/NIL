# datasets

Logical scalar schemas, immutable snapshots and atomic records-plus-offset ingestion.

## Usage

```python
import nil
schema = nil.datasets.Schema({'x': 'float', 'label': {'type': 'string', 'nullable': True}})
with nil.datasets.StreamStore('records.sqlite') as store:
    receipt = store.append('source', 0, [{'x': 1.0}], schema=schema, transaction_id='batch-1')
    print(receipt, store.read('source'))
```

## Parameters

Schema(fields,version=1): int/float/bool/string/datetime, nullable and optional defaults. snapshot/load_snapshot support record/byte bounds. A newer schema may add nullable/defaulted fields or widen int→float. StreamStore.append(stream,start,records,schema,transaction_id) commits records and the next offset together. read bounds count and bytes.

## Output

JSON-compatible records with verified logical schema, immutable JSONL artifacts, or durable SQLite batch receipts/offsets. Timezone-aware datetime values normalize to ISO strings. Replaying an identical transaction ID returns its original receipt; changed payloads or stale offsets fail.

## Requirements

Core NIL and Python standard library only; no numerical backend.

## Limitations

Scalar logical schemas, not Arrow/Parquet or preservation of NumPy/Pandas physical dtypes. Renames, removal, narrowing and nullable→required changes need an explicit external migration. Checksums detect corruption, not source authenticity. SQLite transactions cover this local record sink and offsets, not arbitrary model weights or an external message broker. Stream/source identity is caller-owned; concurrent processes must use the same local database on a reliable local filesystem. Default batch/read budget 8 MB; schema max 32,000 bytes; bounded reads max 10,000 rows. Database rows persist until caller-managed retention. Snapshots and ledgers should be trusted local data.
