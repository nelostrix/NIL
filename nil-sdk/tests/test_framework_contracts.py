"""Public contract checks independent of optional numerical packages."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import nil


def test_inventory_contract_and_public_paths():
    data = nil.coverage.matrix()
    assert data['schema_version'] == 1
    assert data['sdk_version'] == nil.__version__
    assert [area['id'] for area in data['areas']] == list(range(1, 33))
    assert len({r['id'] for r in data['workflows']}) == len(data['workflows'])
    for row in data['workflows']:
        assert row['implementation'] in {'native', 'integration', 'hybrid', 'unimplemented'}
        assert row['validation'] in {'validated', 'target_blocked', 'experimental'}
        assert row['inputs']['description'] and row['outputs']['description'] and row['limits']
        assert all(1 <= n <= 32 for n in row['area_ids'])
        assert set(row['tested_profiles']) <= data['profiles'].keys()
        assert nil.help(row['help'])
        for api in row['api']:
            value = nil
            for part in api.split('.')[1:]:
                value = getattr(value, part)
            assert callable(value), api
    data['workflows'][0]['limits'].clear()
    assert nil.coverage.matrix()['workflows'][0]['limits']
    assert all(a['inventory_status'] in {'partial', 'not_audited'} for a in nil.coverage.areas())


@pytest.mark.parametrize('kwargs', [{'area':True}, {'area':0}, {'area':33}, {'implementation':'magic'}, {'validation':'complete'}])
def test_invalid_inventory_filters(kwargs):
    with pytest.raises(ValueError):nil.coverage.list_workflows(**kwargs)


def test_metadata_is_not_runtime_acceptance(monkeypatch):
    from importlib import metadata
    def missing(name):raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(metadata,'version',missing)
    info = nil.coverage.availability('neural.supervised')
    assert info['dependency_state'] == 'missing' and 'torch' in info['missing']
    assert not info['runtime_verified'] and not info['hardware_verified']
    assert nil.coverage.availability('native.tensor')['dependency_state'] == 'not_required'
    with pytest.raises(ValueError):nil.coverage.describe('unknown.workflow')


def test_passive_discovery_without_site_packages(tmp_path):
    code = '''
import sys, pathlib, threading, json
before=set(pathlib.Path('.').iterdir())
def audit(event,args):
    if event.startswith('socket.') or event in {'subprocess.Popen','os.system'}:raise AssertionError(event)
sys.addaudithook(audit)
import nil
assert len(nil.coverage.areas())==32
nil.coverage.availability('neural.supervised')
import nil.contracts
assert not {'torch','numpy','sklearn','PIL','transformers'} & sys.modules.keys()
assert threading.active_count()==1
assert set(pathlib.Path('.').iterdir())==before
from nil.cli import main
sys.argv=['nil','workflows','native.tensor','--json','--availability']
main()
'''
    env=dict(os.environ, PYTHONPATH=str(Path(nil.__file__).resolve().parent.parent), PYTHONDONTWRITEBYTECODE='1')
    proc=subprocess.run([sys.executable,'-S','-c',code],cwd=tmp_path,env=env,capture_output=True,text=True,check=True)
    rows=json.loads(proc.stdout)
    assert rows[0]['id']=='native.tensor'
    assert rows[0]['availability']['runtime_verified'] is False


def test_result_info_rejects_arbitrary_objects():
    with pytest.raises(TypeError):nil.contracts.result_info(object())
    with pytest.raises(TypeError):nil.contracts.result_info(dict(model=object(),history=[],steps=1))


def test_native_result_metadata():
    model=nil.native.Linear(1,1)
    result=nil.native.fit(model, [[0.],[1.]], [[0.],[2.]], epochs=2)
    info=nil.contracts.result_info(result)
    assert info.family=='native' and info.actual_steps==2 and info.model_count==1
    assert 'model' not in info.to_dict()
    json.dumps(info.to_dict())
