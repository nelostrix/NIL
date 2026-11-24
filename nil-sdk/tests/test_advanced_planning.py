import json
import os
from pathlib import Path
import nil
import pytest
from nil import tool


@pytest.fixture(autouse=True)
def worker_imports(monkeypatch):
    monkeypatch.setenv('PYTHONPATH',str(Path(__file__).parent)+os.pathsep+os.environ.get('PYTHONPATH',''))


@tool
def choose(goal: str, history: list) -> dict:
    if history:return {'final':history[-1]['result']}
    return {'tool':'power','arguments':{'base':2,'exp':3}}


@tool
def power(base: int, exp: int) -> int:
    return pow(base,exp)


@tool
def fail(value: int) -> int:
    raise RuntimeError('External state is ambiguous')


def test_agent_actual_owned_decision_action_resume_and_contract(tmp_path):
    options=dict(policy_version='1',allowed=['power'],tool_versions={'power':'1'},journal_root=tmp_path/'agent',max_calls=2)
    result=nil.planning.agent('calculate',choose,{'power':power},**options)
    assert result['status']=='completed' and result['result']==8 and len(result['history'])==1
    assert nil.planning.agent('calculate',choose,{'power':power},resume=True,**options)==result
    with pytest.raises(ValueError):nil.planning.agent('changed',choose,{'power':power},resume=True,**options)
    with pytest.raises(PermissionError):nil.planning.agent('calculate',choose,{'power':power},**dict(options,allowed=[],journal_root=tmp_path/'denied'))


def test_failed_call_requires_verified_result_and_never_retries(tmp_path):
    path=tmp_path/'plan.json';steps=[dict(id='once',tool='fail',arguments={'value':1})]
    options=dict(allowed=['fail'],tool_versions={'fail':'1'},journal=path)
    with pytest.raises(nil.execution.ExecutionError,match='External state is ambiguous'):nil.planning.run(steps,{'fail':fail},**options)
    assert json.loads(path.read_text())['status']=='failed'
    with pytest.raises(ValueError):nil.planning.reconcile(path,step_id='once',result=7,evidence='')
    nil.planning.reconcile(path,step_id='once',result=7,evidence='Operator verified external output record 123')
    result=nil.planning.run(steps,{'fail':fail},resume=True,**options)
    assert result['completed']==[{'id':'once','result':7}]
    assert result['reconciliations'][0]['evidence'].endswith('123')
