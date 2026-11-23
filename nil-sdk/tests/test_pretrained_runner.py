"""Acceptance preparation refuses drift, unknown budgets and invalid snapshots."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]


def runner():
    spec=importlib.util.spec_from_file_location('nil_pretrained_runner',ROOT/'scripts/accept_pretrained.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_cached_snapshot_is_verified_without_contacting_hub(tmp_path,monkeypatch):
    tool=runner();payload=tmp_path/'model.safetensors';payload.write_bytes(b'fixture-not-a-real-model')
    manifest={'repo_id':'fixture/model','revision':'a'*40,'download_bytes':payload.stat().st_size,
              'files_sha256':{payload.name:hashlib.sha256(payload.read_bytes()).hexdigest()}}
    (tmp_path/'nil-source.json').write_text(json.dumps(manifest))
    monkeypatch.setitem(sys.modules,'huggingface_hub',None)
    assert tool.prepare('fixture/model',tmp_path,'a'*40,1024)==manifest
    with pytest.raises(FileExistsError):tool.prepare('other/model',tmp_path,'a'*40,1024)
    with pytest.raises(ValueError,match='budget'):tool.prepare('fixture/model',tmp_path,'a'*40,1)
    payload.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='integrity'):tool.prepare('fixture/model',tmp_path,'a'*40,1024)


def test_preparation_requires_immutable_revision_and_known_download_budget(tmp_path,monkeypatch):
    tool=runner()
    with pytest.raises(ValueError,match='immutable'):tool.prepare('fixture/model',tmp_path,'main',1024)
    info=SimpleNamespace(sha='a'*40,siblings=[SimpleNamespace(rfilename='model.safetensors',size=2048)])
    hub=SimpleNamespace(HfApi=lambda:SimpleNamespace(model_info=lambda *a,**k:info),snapshot_download=lambda *a,**k:pytest.fail('Download must not start'))
    monkeypatch.setitem(sys.modules,'huggingface_hub',hub)
    with pytest.raises(ValueError,match='budget'):tool.prepare('fixture/model',tmp_path/'new','a'*40,1024)
    assert not (tmp_path/'new').exists()


def test_snapshot_integrity_failure_publishes_failed_report_without_workers(tmp_path):
    folder=tmp_path/'snapshot';folder.mkdir();(folder/'model.safetensors').write_bytes(b'changed')
    (folder/'nil-source.json').write_text(json.dumps({'repo_id':'fixture/model','revision':'a'*40,
        'download_bytes':7,'files_sha256':{'model.safetensors':'0'*64}}))
    report,code=runner().run(folder,tmp_path/'result.json',['inference'],False)
    assert code==2 and report['status']=='failed' and report['stages']=={}
    assert 'digest differs' in report['error']
