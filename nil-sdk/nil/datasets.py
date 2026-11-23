"""Typed JSONL snapshots, additive schema migration and transactional local ingestion."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from nil._classical_common import positive_int


def _json(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


class Schema:
    def __init__(self, fields, *, version=1):
        positive_int(version,'version')
        if not isinstance(fields,dict) or not 0<len(fields)<=256: raise ValueError('Schema needs 1–256 named fields')
        self.version,self.fields=version,{}
        for name,definition in fields.items():
            if not isinstance(name,str) or not name or len(name)>128: raise ValueError('Invalid field name')
            spec={'type':definition} if isinstance(definition,str) else dict(definition)
            if set(spec)-{'type','nullable','default'} or spec.get('type') not in {'int','float','bool','string','datetime'} or type(spec.get('nullable',False)) is not bool: raise ValueError('Invalid field specification')
            spec.setdefault('nullable',False); self.fields[name]=spec
            if 'default' in spec: spec['default']=self._value(spec['default'],spec)
        if len(_json(self.to_dict()).encode())>32000: raise ValueError('Schema exceeds 32 KiB')

    def _value(self,value,spec):
        if value is None:
            if not spec['nullable']: raise ValueError('Null in a nonnullable field')
            return None
        kind=spec['type']
        if kind=='int' and type(value) is int: return value
        if kind=='float' and type(value) in {int,float} and math.isfinite(value): return float(value)
        if kind=='bool' and type(value) is bool: return value
        if kind=='string' and type(value) is str: return value
        if kind=='datetime' and isinstance(value,(str,datetime)):
            parsed=datetime.fromisoformat(value) if isinstance(value,str) else value
            if parsed.tzinfo is None or parsed.utcoffset() is None: raise ValueError('Datetime values need an explicit timezone')
            return parsed.isoformat()
        raise ValueError(f'Value does not match {kind}')

    def validate(self,record):
        if not isinstance(record,dict) or set(record)-set(self.fields): raise ValueError('Record has unknown fields')
        result={}
        for name,spec in self.fields.items():
            if name in record: value=record[name]
            elif 'default' in spec: value=deepcopy(spec['default'])
            elif spec['nullable']: value=None
            else: raise ValueError(f'Missing required field {name}')
            result[name]=self._value(value,spec)
        return result

    def to_dict(self): return {'version':self.version,'fields':deepcopy(self.fields)}

    def accepts(self,previous):
        if self.to_dict()==previous.to_dict(): return
        if self.version<=previous.version: raise ValueError('Schema changes require an increased version')
        for name,spec in previous.fields.items():
            new=self.fields.get(name)
            if new is None or not (new['type']==spec['type'] or (spec['type'],new['type'])==('int','float')) or spec['nullable'] and not new['nullable']:
                raise ValueError('Schema migration cannot remove, narrow or rename fields')
        for name,spec in self.fields.items():
            if name not in previous.fields and not spec['nullable'] and 'default' not in spec:
                raise ValueError('Added fields need nullable=True or a default')


def _schema(value):
    if not isinstance(value,Schema): raise TypeError('Pass a nil.datasets.Schema')
    return value


def snapshot(records,path,*,schema,max_records=1000000,max_bytes=100000000):
    import os,shutil,tempfile
    from nil.data import snapshot as publish
    schema=_schema(schema); destination=Path(path).absolute()
    if destination.exists() or destination.is_symlink(): raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.nil-typed-',dir=destination.parent) as temp:
        staged=publish((schema.validate(row) for row in records),Path(temp)/'payload',max_records=max_records,max_bytes=max_bytes)
        manifest=json.loads((staged/'manifest.json').read_text())
        manifest['typed_schema']=schema.to_dict()
        manifest['schema_sha256']=hashlib.sha256(_json(schema.to_dict()).encode()).hexdigest()
        (staged/'manifest.json').write_text(_json(manifest))
        destination.mkdir()
        try:
            os.replace(staged/'records.jsonl',destination/'records.jsonl')
            os.replace(staged/'manifest.json',destination/'manifest.json')
        except BaseException:
            shutil.rmtree(destination); raise
    return destination


def load_snapshot(path,*,schema=None,max_records=1000000,max_bytes=100000000):
    from nil.data import load_snapshot as read,Records
    root=Path(path); metadata=root/'manifest.json'
    if root.is_symlink() or metadata.is_symlink() or metadata.stat().st_size>65536: raise ValueError('Invalid typed manifest')
    manifest=json.loads(metadata.read_text()); definition=manifest.get('typed_schema')
    if not isinstance(definition,dict) or hashlib.sha256(_json(definition).encode()).hexdigest()!=manifest.get('schema_sha256'): raise ValueError('Typed schema integrity check failed')
    original=Schema(definition['fields'],version=definition['version'])
    desired=original if schema is None else _schema(schema)
    desired.accepts(original)
    return Records(desired.validate(original.validate(row)) for row in read(path,max_records=max_records,max_bytes=max_bytes).records)


class StreamStore:
    """Records and source offsets commit together; arbitrary model state is not part of the transaction."""
    def __init__(self,path):
        from nil._ledger import connect
        self._db=connect(path)
        self._db.executescript('''
CREATE TABLE IF NOT EXISTS streams(name TEXT PRIMARY KEY,offset INTEGER NOT NULL,schema_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS batches(stream TEXT NOT NULL,token TEXT NOT NULL,digest TEXT NOT NULL,receipt TEXT NOT NULL,PRIMARY KEY(stream,token));
CREATE TABLE IF NOT EXISTS records(stream TEXT NOT NULL,position INTEGER NOT NULL,payload TEXT NOT NULL,schema_json TEXT NOT NULL,PRIMARY KEY(stream,position));
''')

    def offset(self,stream):
        row=self._db.execute('SELECT offset FROM streams WHERE name=?',(stream,)).fetchone()
        return row[0] if row else 0

    def append(self,stream,start,records,*,schema,transaction_id,max_records=1000,max_bytes=8000000):
        import itertools
        positive_int(max_records,'max_records'); positive_int(max_bytes,'max_bytes'); schema=_schema(schema)
        if not isinstance(stream,str) or not stream or len(stream)>128 or not isinstance(transaction_id,str) or not transaction_id or len(transaction_id)>128: raise ValueError('Stream and transaction IDs need 1–128 characters')
        if type(start) is not int or start<0: raise ValueError('start must be a nonnegative offset')
        payload=[]; size=0
        for row in itertools.islice(records,max_records+1):
            encoded=_json(schema.validate(row)); size+=len(encoded.encode())
            if len(payload)>=max_records or size>max_bytes: raise ValueError('Batch exceeds record/byte budget')
            payload.append(encoded)
        if not payload: raise ValueError('Empty batches are not committed')
        definition=_json(schema.to_dict()); digest=hashlib.sha256(_json([start,definition,payload]).encode()).hexdigest()
        self._db.execute('BEGIN IMMEDIATE')
        try:
            previous=self._db.execute('SELECT digest,receipt FROM batches WHERE stream=? AND token=?',(stream,transaction_id)).fetchone()
            if previous:
                if previous[0]!=digest: raise ValueError('Transaction ID reused with different data')
                self._db.execute('COMMIT'); return json.loads(previous[1])
            current=self._db.execute('SELECT offset,schema_json FROM streams WHERE name=?',(stream,)).fetchone()
            if start!=(current[0] if current else 0): raise ValueError('Source offset conflict; reload committed offset')
            if current:
                old=json.loads(current[1]); schema.accepts(Schema(old['fields'],version=old['version']))
            for i,encoded in enumerate(payload): self._db.execute('INSERT INTO records VALUES(?,?,?,?)',(stream,start+i,encoded,definition))
            receipt={'stream':stream,'start':start,'next_offset':start+len(payload),'count':len(payload)}
            self._db.execute('INSERT INTO streams VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET offset=excluded.offset,schema_json=excluded.schema_json',(stream,receipt['next_offset'],definition))
            self._db.execute('INSERT INTO batches VALUES(?,?,?,?)',(stream,transaction_id,digest,_json(receipt)))
            self._db.execute('COMMIT'); return receipt
        except BaseException:
            self._db.execute('ROLLBACK'); raise

    def read(self,stream,*,start=0,limit=1000,schema=None,max_bytes=8000000):
        positive_int(limit,'limit')
        if limit>10000 or type(start) is not int or start<0: raise ValueError('Invalid read offset or limit exceeds 10000')
        positive_int(max_bytes,'max_bytes'); result=[]; size=0
        for payload,definition in self._db.execute('SELECT payload,schema_json FROM records WHERE stream=? AND position>=? ORDER BY position LIMIT ?',(stream,start,limit)):
            size+=len(payload.encode())
            if size>max_bytes: raise ValueError('Read exceeds max_bytes')
            record=json.loads(payload)
            if schema is not None:
                _schema(schema); old=json.loads(definition); schema.accepts(Schema(old['fields'],version=old['version'])); record=schema.validate(record)
            result.append(record)
        return result

    def close(self): self._db.close()
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
