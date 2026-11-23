"""Real Linux process budgets, and explicit opt-in kernel containment acceptance."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest
from nil import execution
from nil._execution_resources import group_rss,validate_cgroup_root

linux=pytest.mark.skipif(sys.platform!='linux',reason='Linux resource containment')


def test_resource_options_preflight_and_capabilities(tmp_path):
    for options in ({'group_rss_mb':0},{'group_rss_mb':True},{'aggregate_memory_mb':-1},
                    {'aggregate_memory_mb':128},{'cgroup_root':str(tmp_path)}):
        with pytest.raises(ValueError): execution.Limits(**options)
    if sys.platform=='linux':
        with pytest.raises(execution.ResourceUnavailable,match='cgroup-v2 mount'):
            execution.submit('builtins:sum',[1],limits=execution.Limits(aggregate_memory_mb=128,cgroup_root=tmp_path),job_root=tmp_path/'jobs')
        assert not (tmp_path/'jobs').exists()
    report=execution.capabilities(cgroup_root=tmp_path)
    assert report['aggregate_rss_limit'] is False and report['gpu_memory_limit'] is False
    assert report['kernel_aggregate_memory']['available'] is False


@linux
def test_live_process_group_rss_reports_actual_members():
    report=group_rss(os.getpgrp())
    assert report['rss_bytes']>0 and report['processes']>=1


@linux
def test_rss_limit_accounts_for_children_and_cleans_them_up(tmp_path):
    child_file=tmp_path/'child.pid'; evidence=tmp_path/'rss.json'
    child_code="import os,sys,time; data=bytearray(48*1024**2); open(sys.argv[1],'w').write(str(os.getpid())); time.sleep(30)"
    code='''
import json,os,subprocess,sys,time
from pathlib import Path
from nil._execution_resources import group_rss
child=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]])
deadline=time.monotonic()+5
while not Path(sys.argv[2]).exists():
    if time.monotonic()>deadline: raise RuntimeError('child startup failed')
    time.sleep(.005)
# Each allocation is below the group budget; the combined usage exceeds it.
data=bytearray(48*1024**2)
Path(sys.argv[3]).write_text(json.dumps(group_rss(os.getpgrp())))
time.sleep(30)
'''
    job=execution.submit('subprocess:run',[sys.executable,'-c',code,child_code,str(child_file),str(evidence)],
        limits=execution.Limits(timeout=12,group_rss_mb=96),job_root=tmp_path/'jobs',check=True)
    with pytest.raises(execution.ExecutionResourceLimit,match='RSS budget exceeded'): job.result()
    record=job.status(); assert record['status']=='resource_limited'
    assert record['group_memory']['sampled_peak_rss_bytes']>96*1024**2
    assert record['group_memory']['sampled_peak_processes']>=3
    assert execution._identity(int(child_file.read_text())) is None
    assert not (job.path/'input.pkl').exists() and not (job.path/'result.pkl').exists()
    assert execution.run('builtins:sum',[1,2],job_root=tmp_path/'healthy')==3


@linux
def test_normal_rss_budget_records_peak_and_cancellation_still_works(tmp_path):
    job=execution.submit('time:sleep',.15,limits=execution.Limits(group_rss_mb=128),job_root=tmp_path/'success')
    assert job.result() is None
    assert 0<job.status()['group_memory']['sampled_peak_rss_bytes']<128*1024**2
    job=execution.submit('time:sleep',30,limits=execution.Limits(group_rss_mb=128),job_root=tmp_path/'cancel')
    job.cancel()
    with pytest.raises(execution.ExecutionCancelled): job.result()


@linux
def test_rss_guard_survives_actual_client_kill(tmp_path):
    code='''
import sys,time
from nil import execution
job=execution.submit('subprocess:run',[sys.executable,'-c',"import time; time.sleep(.4); data=bytearray(80*1024**2); time.sleep(30)"],
    limits=execution.Limits(timeout=10,group_rss_mb=64),job_root=sys.argv[1],check=True)
print(job.path,flush=True)
time.sleep(30)
'''
    client=subprocess.Popen([sys.executable,'-c',code,str(tmp_path)],stdout=subprocess.PIPE,text=True)
    try:
        job=Path(client.stdout.readline().strip()); client.kill(); client.wait(timeout=3)
        deadline=time.monotonic()+12
        while time.monotonic()<deadline:
            record=json.loads((job/'job.json').read_text())
            if record['status']=='resource_limited': break
            time.sleep(.02)
        assert record['status']=='resource_limited'
        assert execution._identity(record['pid']) is None
    finally:
        if client.poll() is None: client.kill(); client.wait(timeout=3)
        client.stdout.close()


def _delegated_root():
    root=os.environ.get('NIL_TEST_CGROUP_ROOT')
    if root is None: pytest.skip('Real kernel gate requires explicit NIL_TEST_CGROUP_ROOT delegation')
    validate_cgroup_root(root)  # Explicitly requested but invalid: fail, never skip.
    return root


@linux
def test_actual_cgroup_combined_oom_and_cleanup(tmp_path):
    root=_delegated_root()
    code="import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c','import time; x=bytearray(80*1024**2); time.sleep(30)']); x=bytearray(80*1024**2); time.sleep(30)"
    job=execution.submit('subprocess:run',[sys.executable,'-c',code],check=True,
        limits=execution.Limits(timeout=15,aggregate_memory_mb=128,cgroup_root=root),job_root=tmp_path)
    with pytest.raises(execution.ExecutionResourceLimit,match='cgroup OOM'): job.result()
    record=job.status()
    assert record['aggregate_memory']['memory_limit_bytes']==128*1024**2
    assert record['aggregate_memory']['memory_events']['oom_kill']>=1
    assert not Path(record['cgroup_path']).exists()


@linux
def test_actual_cgroup_contains_new_session_descendant(tmp_path):
    root=_delegated_root(); marker=tmp_path/'escape.pid'
    code="import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); open(sys.argv[1],'w').write(str(child.pid)); time.sleep(30)"
    job=execution.submit('subprocess:run',[sys.executable,'-c',code,str(marker)],check=True,
        limits=execution.Limits(timeout=2,aggregate_memory_mb=128,cgroup_root=root),job_root=tmp_path/'jobs')
    with pytest.raises(execution.ExecutionTimeout): job.result()
    assert execution._identity(int(marker.read_text())) is None
    assert not Path(job.status()['cgroup_path']).exists()


@linux
def test_resource_limit_reconciles_registered_run(tmp_path):
    code="from nil.lifecycle import Run; import sys,time; r=Run(sys.argv[1],'memory-budget'); r.status('running'); x=bytearray(80*1024**2); time.sleep(30)"
    job=execution.submit('subprocess:run',[sys.executable,'-c',code,str(tmp_path/'runs')],check=True,
        limits=execution.Limits(timeout=10,group_rss_mb=64),job_root=tmp_path/'jobs')
    with pytest.raises(execution.ExecutionResourceLimit): job.result()
    (run_file,)=(tmp_path/'runs').glob('*/run.json')
    record=json.loads(run_file.read_text())
    assert record['status']=='failed' and record['execution_status']=='resource_limited'
