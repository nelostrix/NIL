"""Shared immutable JSON snapshots for bounded opt-in workflows."""
from nil.data import snapshot, load_snapshot


def save(path, kind, value):
    return snapshot([{'schema_version': 1, 'kind': kind, 'value': value}], path,
                    max_records=1, max_bytes=16*1024*1024)


def load(path, kind):
    rows = load_snapshot(path, max_records=1, max_bytes=16*1024*1024)
    row = rows[0]
    if not isinstance(row, dict) or set(row) != {'schema_version','kind','value'} or type(row['schema_version']) is not int or row['schema_version'] != 1 or row['kind'] != kind:
        raise ValueError('Unsupported workflow artifact')
    return row['value']
