"""Backend-free retrieval, probabilistic and owned-tool workflow contracts."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import nil
from nil.tools.base import Tool


def test_rag_retrieval_updates_citations_and_restart(tmp_path):
    index=nil.rag.Index(chunk_chars=100,overlap=10)
    index.add('install','Installing the NIL base package installs only nil-sdk. No backend is automatically installed.',source='installation.md')
    index.add('gpu','GPU training requires an explicitly installed backend and validated hardware.',source='hardware.md')
    hits=index.search('GPU training')
    assert hits[0]['document_id']=='gpu' and hits[0]['source']=='hardware.md'
    assert index.ask('GPU')['citations']==['gpu:0']
    assert index.ask('nonexistentword')['abstained']
    assert index.ask('GPU',allowed_documents=['install'])['abstained']
    result=nil.rag.evaluate(index,[dict(question='GPU training',relevant_documents=['gpu'])],k=1)
    assert result['recall@1']==1 and result['mrr']==1
    artifact=index.save(tmp_path/'index');restored=nil.rag.Index.load(artifact)
    assert restored.search('GPU training')==hits
    with pytest.raises(FileExistsError):index.save(artifact)
    restored.add('gpu','CUDA accelerator setup',source='new.md')
    assert restored.ask('GPU')['abstained']
    restored.delete('gpu');assert restored.ask('CUDA')['abstained']
    with pytest.raises(ValueError,match='citations'):
        index.ask('GPU',generator=lambda question,hits:dict(text='invented',citations=['wrong']))
    text=index._documents['gpu']['text']
    before=index.search('GPU')
    index.config['max_chars']=1
    with pytest.raises(ValueError,match='budget'):index.add('gpu',text)
    assert index.search('GPU')==before


def test_rag_integrity_rejects_corrupt_snapshot(tmp_path):
    path=nil.rag.Index().add('doc','Actual local document').save(tmp_path/'index')
    (path/'records.jsonl').write_text('corrupt')
    with pytest.raises(ValueError,match='integrity|bytes|limits'):nil.rag.Index.load(path)


def test_beta_conjugate_update_and_atomic_failure(tmp_path):
    model=nil.probabilistic.BetaBernoulli(2,3).update([1,0,1])
    assert model.predict()['probability']==.5
    assert model.predict()['variance']==pytest.approx(16/(64*9))
    before=model.predict()
    with pytest.raises(ValueError):model.update([1,2])
    assert model.predict()==before
    assert nil.probabilistic.BetaBernoulli.load(model.save(tmp_path/'posterior')).predict()==before


def test_shortest_path_and_budget():
    graph={'a':['c','b'],'b':['d'],'c':['d']}
    assert nil.planning.shortest_path('a','d',graph)==['a','b','d']
    assert nil.planning.shortest_path('a','missing',graph)==[]
    with pytest.raises(ValueError,match='budget'):nil.planning.shortest_path('a','d',graph,max_states=2)


@pytest.fixture
def tool_module(tmp_path,monkeypatch):
    source='''from nil import tool
import time
@tool
def add(a: int, b: int) -> int:
    return a+b
@tool
def pause(seconds: float) -> int:
    time.sleep(seconds)
    return 1
'''
    name='nil_test_plan_tools'
    (tmp_path/(name+'.py')).write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PYTHONPATH',str(tmp_path)+os.pathsep+os.environ.get('PYTHONPATH',''))
    sys.modules.pop(name,None)
    return importlib.import_module(name)


def test_owned_plan_references_recovery_and_permissions(tool_module,tmp_path):
    steps=[dict(id='one',tool='add',arguments=dict(a=2,b=3)),dict(id='two',tool='add',arguments=dict(a={'$ref':'one'},b=4))]
    tools={'add':tool_module.add};options=dict(allowed=['add'],journal=tmp_path/'trace.json',tool_versions={'add':'1'})
    result=nil.planning.run(steps,tools,**options)
    assert [row['result'] for row in result['completed']]==[5,9]
    assert nil.planning.run(steps,tools,resume=True,**options)==result
    with pytest.raises(ValueError,match='mismatch'):
        nil.planning.run(steps,tools,resume=True,**dict(options,tool_versions={'add':'2'}))
    with pytest.raises(PermissionError):nil.planning.run(steps,tools,**dict(options,allowed=[]))
    broken=[dict(id='bad',tool='add',arguments=dict(a={'$ref':'future'},b=1))]
    with pytest.raises(ValueError,match='prior'):nil.planning.run(broken,tools,**dict(options,journal=tmp_path/'bad.json'))
    assert not (tmp_path/'bad.json').exists()
    result['status']='running';(tmp_path/'trace.json').write_text(json.dumps(result))
    with pytest.raises(RuntimeError,match='Reconcile'):nil.planning.run(steps,tools,resume=True,**options)


def test_owned_plan_deadline_is_terminal(tool_module,tmp_path):
    options=dict(allowed=['pause'],journal=tmp_path/'trace.json',tool_versions={'pause':'1'},timeout=.3)
    steps=[dict(id='sleep',tool='pause',arguments=dict(seconds=5.))]
    with pytest.raises(nil.execution.ExecutionTimeout):nil.planning.run(steps,{'pause':tool_module.pause},**options)
    assert json.loads((tmp_path/'trace.json').read_text())['status']=='failed'
    with pytest.raises(RuntimeError,match='Reconcile'):nil.planning.run(steps,{'pause':tool_module.pause},resume=True,**options)


def test_seven_modules_import_without_backends_or_workers(tmp_path):
    source=str(Path(nil.__file__).resolve().parent.parent)
    code="""import sys,threading
import nil
for name in ('rag','planning','probabilistic','learning','audio','multimodal','graphs'):getattr(nil,name)
assert not {'torch','numpy','PIL','sklearn'} & sys.modules.keys()
assert threading.active_count()==1
"""
    subprocess.run([sys.executable,'-S','-c',code],env={**os.environ,'PYTHONPATH':source},cwd=tmp_path,check=True)


def test_bm25_matches_independent_score():
    import math
    index=nil.rag.Index().add('a','x x y').add('b','x y y z')
    row=index.search('x')[0]
    expected=math.log(1+.5/2.5)*2*2.2/(2+1.2*(.25+.75*3/3.5))
    assert row['document_id']=='a' and row['score']==pytest.approx(expected)


def test_plan_cancellation_before_launch_is_resumable(tool_module,tmp_path):
    import threading
    event=threading.Event();event.set()
    plan=[dict(id='one',tool='add',arguments=dict(a=1,b=2))]
    kwargs=dict(allowed=['add'],journal=tmp_path/'cancel.json',tool_versions={'add':'1'},cancel_event=event)
    cancelled=nil.planning.run(plan,{'add':tool_module.add},**kwargs)
    assert cancelled['status']=='cancelled' and not cancelled['completed']
    assert not (tmp_path/'tool_jobs').exists()
    event.clear()
    assert nil.planning.run(plan,{'add':tool_module.add},resume=True,**kwargs)['completed'][0]['result']==3


def test_dead_supervisor_recovers_or_refuses_safely_without_hanging(tmp_path,monkeypatch):
    import time
    original=nil.execution.submit
    def terminate_before_start(*args,**kwargs):
        job=original(*args,**kwargs)
        job.process.kill();job.process.wait()
        return job
    monkeypatch.setattr(nil.execution,'submit',terminate_before_start)
    plan=[dict(id='one',tool='power',arguments=dict(base=2,exp=3))]
    tool=Tool(name='power',parameters={'base':{'type':'integer'},'exp':{'type':'integer'}},fn=pow)
    started=time.monotonic()
    expected=nil.execution.ExecutionError if nil.execution.capabilities()['identity_safe_recovery'] else NotImplementedError
    with pytest.raises(expected):
        nil.planning.run(plan,{'power':tool},allowed=['power'],journal=tmp_path/'trace.json',tool_versions={'power':'1'})
    assert time.monotonic()-started<5
    assert json.loads((tmp_path/'trace.json').read_text())['status']=='failed'


@pytest.mark.parametrize('alpha,beta',[(1e-200,1e-200),(1e12,1e12),(1e-200,1.)])
def test_beta_extreme_valid_priors_remain_finite(alpha,beta):
    import math
    result=nil.probabilistic.BetaBernoulli(alpha,beta).predict()
    assert math.isfinite(result['probability']) and math.isfinite(result['variance'])
    assert 0<=result['probability']<=1 and 0<=result['variance']<=.25
    assert result['variance']==nil.probabilistic.BetaBernoulli(beta,alpha).predict()['variance']
