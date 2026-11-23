"""Acceptance tools must preserve failures rather than turn missing checks green."""
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
pytestmark=pytest.mark.skipif(sys.platform!='linux',reason='Linux fresh-process benchmark accounting')


def call(script,*args):
    return subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],capture_output=True,text=True,timeout=30)


def test_core_benchmark_reports_samples_and_budget_failures(tmp_path):
    output=tmp_path/'pass.json'
    result=call('benchmark_workloads.py','--profile','core','--size','small','--repeats',1,'--output',output)
    assert result.returncode==0,result.stderr
    report=json.loads(output.read_text()); assert report['status']=='passed'
    assert not report['old_pc_acceptance']
    assert all(not row['optional_backends_loaded'] for case in report['cases'].values() for row in case['samples'])
    budgets=tmp_path/'budgets.json'; budgets.write_text(json.dumps({'core_import':{'max_wall_s':1e-12}}))
    result=call('benchmark_workloads.py','--size','small','--repeats',1,'--budgets',budgets,'--output',output)
    assert result.returncode==1
    report=json.loads(output.read_text()); assert report['status']=='failed'
    assert report['cases']['core_import']['budget_violations'][0]['metric']=='max_wall_s'


def test_benchmark_timeout_retains_failed_samples(tmp_path):
    output=tmp_path/'timeout.json'
    result=call('benchmark_workloads.py','--size','small','--repeats',1,'--timeout',.000001,'--output',output)
    assert result.returncode==1
    report=json.loads(output.read_text())
    for case in report['cases'].values():
        assert case['status']=='failed' and not case['samples']
        assert case['failures'][0]['error']=='TimeoutExpired'


def test_unavailable_kernel_acceptance_fails_before_checks(tmp_path):
    output=tmp_path/'unavailable.json'
    result=call('accept_execution.py','--cgroup-root',tmp_path,'--output',output)
    assert result.returncode==2,result.stderr
    report=json.loads(output.read_text())
    assert report['status']=='failed' and report['checks']==[]
    assert 'Required capability unavailable' in report['error']


def test_invalid_budgets_are_rejected_before_workloads(tmp_path):
    bad=tmp_path/'bad.json'; bad.write_text('{"core_import":{"max_wall_s":NaN}}')
    result=call('benchmark_workloads.py','--budgets',bad,'--output',tmp_path/'output.json')
    assert result.returncode==2 and not (tmp_path/'output.json').exists()


def test_gpu_gate_refuses_absent_hardware_instead_of_passing_skips(tmp_path):
    try:
        import torch
        if torch.cuda.is_available(): pytest.skip('This regression tests actual GPU absence')
    except ImportError: pass
    output=tmp_path/'gpu.json'
    result=call('accept_gpu.py','--mode','all','--output',output)
    assert result.returncode==2,result.stderr
    report=json.loads(output.read_text())
    assert report['status']=='failed' and report['tests']=={} and report['error']
