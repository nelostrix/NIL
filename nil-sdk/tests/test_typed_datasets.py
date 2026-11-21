import json,os,subprocess,sys,time
from pathlib import Path
import pytest
import nil


def test_typed_snapshot_schema_evolution_integrity_and_failures(tmp_path):
    original=nil.datasets.Schema({'x':'int','name':{'type':'string','nullable':True}})
    root=nil.datasets.snapshot([{'x':1},{'x':2,'name':'second'}],tmp_path/'data',schema=original)
    evolved=nil.datasets.Schema({'x':'float','name':{'type':'string','nullable':True},'active':{'type':'bool','default':True}},version=2)
    assert nil.datasets.load_snapshot(root,schema=evolved).records==[{'x':1.,'name':None,'active':True},{'x':2.,'name':'second','active':True}]
    with pytest.raises(ValueError): nil.datasets.snapshot([{'x':True}],tmp_path/'bad',schema=original)
    assert not (tmp_path/'bad').exists()
    with pytest.raises(ValueError): nil.datasets.Schema({'x':'int'},version=2).accepts(original)
    manifest=root/'manifest.json'; values=json.loads(manifest.read_text()); values['typed_schema']['fields']['x']['type']='float'; manifest.write_text(json.dumps(values))
    with pytest.raises(ValueError,match='integrity'): nil.datasets.load_snapshot(root)


def test_transactional_source_offsets_idempotency_schema_and_atomic_failure(tmp_path):
    schema=nil.datasets.Schema({'x':'int'}); path=tmp_path/'stream.db'
    with nil.datasets.StreamStore(path) as store:
        first=store.append('source',0,[{'x':1},{'x':2}],schema=schema,transaction_id='first')
        assert first['next_offset']==2
        assert store.append('source',0,[{'x':1},{'x':2}],schema=schema,transaction_id='first')==first
        with pytest.raises(ValueError): store.append('source',0,[{'x':3}],schema=schema,transaction_id='second')
        with pytest.raises(ValueError): store.append('source',2,[{'x':3},{'x':'bad'}],schema=schema,transaction_id='invalid')
        assert store.offset('source')==2 and store.read('source')==[{'x':1},{'x':2}]
        evolved=nil.datasets.Schema({'x':'float','tag':{'type':'string','nullable':True}},version=2)
        store.append('source',2,[{'x':3.5}],schema=evolved,transaction_id='second')
        assert store.read('source',schema=evolved)==[{'x':1.,'tag':None},{'x':2.,'tag':None},{'x':3.5,'tag':None}]
    with nil.datasets.StreamStore(path) as reopened: assert reopened.offset('source')==3


def test_actual_killed_sqlite_writer_cannot_commit_half_a_batch(tmp_path):
    path=tmp_path/'stream.db'; schema=nil.datasets.Schema({'x':'int'})
    with nil.datasets.StreamStore(path) as store: store.append('s',0,[{'x':1}],schema=schema,transaction_id='first')
    code='''
import sys,time
from pathlib import Path
from nil.datasets import StreamStore
store=StreamStore(sys.argv[1]); store._db.execute('BEGIN IMMEDIATE')
store._db.execute("UPDATE streams SET offset=999 WHERE name='s'")
Path(sys.argv[2]).write_text('ready'); time.sleep(60)
'''
    ready=tmp_path/'ready'; child=subprocess.Popen([sys.executable,'-c',code,str(path),str(ready)])
    try:
        deadline=time.monotonic()+5
        while not ready.exists() and time.monotonic()<deadline: time.sleep(.02)
        assert ready.exists(); child.kill(); child.wait(timeout=5)
        with nil.datasets.StreamStore(path) as store: assert store.offset('s')==1 and store.read('s')==[{'x':1}]
    finally:
        if child.poll() is None: child.kill(); child.wait()


def test_datetime_types_and_byte_limits_are_explicit(tmp_path):
    from datetime import datetime,timezone
    schema=nil.datasets.Schema({'time':'datetime','score':'float'})
    assert schema.validate({'time':datetime(2020,1,1,tzinfo=timezone.utc),'score':1})=={'time':'2020-01-01T00:00:00+00:00','score':1.}
    with pytest.raises(ValueError,match='timezone'): schema.validate({'time':'2020-01-01T00:00:00','score':1})
    with nil.datasets.StreamStore(tmp_path/'data.db') as store:
        with pytest.raises(ValueError,match='budget'): store.append('s',0,[{'time':'2020-01-01T00:00:00+00:00','score':1}],schema=schema,transaction_id='bad',max_bytes=1)
        assert store.offset('s')==0
