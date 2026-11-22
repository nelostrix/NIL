"""Actual Windows kernel tests. Linux skips never establish Windows acceptance."""
import json,os,subprocess,sys,time
from pathlib import Path
import pytest
from nil import execution
pytestmark=pytest.mark.skipif(os.name!='nt',reason='Actual Windows Job Object acceptance')


def wait_record(path,status,seconds=15):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        record=json.loads((path/'job.json').read_text())
        if record['status']==status:return record
        time.sleep(.02)
    raise AssertionError(f'Expected {status}; got {record}')


def tree_job(tmp_path,*,success=False):
    child="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],creationflags=subprocess.CREATE_NEW_PROCESS_GROUP); open(sys.argv[1],'w').write(str(p.pid)); time.sleep(60)"
    code="""
import subprocess,sys,time
from pathlib import Path
p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]],creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
Path(sys.argv[3]).write_text(str(p.pid))
deadline=time.monotonic()+8
while not Path(sys.argv[2]).exists():
    if time.monotonic()>deadline: raise RuntimeError('grandchild startup failed')
    time.sleep(.01)
"""+('' if success else '\ntime.sleep(60)')
    grandchild=tmp_path/'grandchild';parent=tmp_path/'child'
    job=execution.submit('subprocess:run',[sys.executable,'-c',code,child,str(grandchild),str(parent)],check=True,
        limits=execution.Limits(timeout=12 if not success else 15),job_root=tmp_path/'jobs')
    return job,parent,grandchild


def assert_tree_dead(parent,grandchild):
    assert parent.exists() and grandchild.exists()
    for marker in (parent,grandchild):assert execution._identity(int(marker.read_text())) is None


def test_windows_success_failure_unicode_paths_and_bounded_output(tmp_path):
    root=tmp_path/'space and unicode Ω'
    job=execution.submit('builtins:sum',[1,2,3],job_root=root)
    assert job.result()==6
    assert job.status()['containment']=='windows-job-object' and job.status()['containment_cleanup']=='closed'
    assert not (job.path/'input.pkl').exists()
    bad=execution.submit('builtins:int','bad',job_root=root)
    with pytest.raises(execution.ExecutionError,match='ValueError'):bad.result()
    noisy=execution.submit('builtins:print','x'*200000,limits=execution.Limits(log_bytes=1024),job_root=root)
    assert noisy.result() is None and (noisy.path/'worker.log').stat().st_size==1024
    large=execution.submit('builtins:bytes',2*1024**2,limits=execution.Limits(payload_mb=1),job_root=root)
    with pytest.raises(execution.ExecutionError,match='payload_mb'):large.result()


@pytest.mark.parametrize('action',['timeout','cancel','success'])
def test_windows_children_and_grandchildren_are_owned_across_process_groups(tmp_path,action):
    job,parent,grandchild=tree_job(tmp_path,success=action=='success')
    if action=='cancel':
        deadline=time.monotonic()+5
        while not grandchild.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert grandchild.exists();job.cancel()
    if action=='success':job.result()
    else:
        with pytest.raises(execution.ExecutionTimeout if action=='timeout' else execution.ExecutionCancelled):job.result()
    assert_tree_dead(parent,grandchild)
    assert job.status()['containment_cleanup']=='closed'


@pytest.mark.parametrize('kill_owner',['client','supervisor'])
def test_windows_owner_death_still_cleans_owned_tree(tmp_path,kill_owner):
    marker=tmp_path/'descendant'
    code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); open(sys.argv[1],'w').write(str(p.pid)); time.sleep(60)"
    client_code="from nil import execution; import sys,time; j=execution.submit('subprocess:run',[sys.executable,'-c',sys.argv[2],sys.argv[3]],check=True,limits=execution.Limits(timeout=7),job_root=sys.argv[1]); print(j.path,flush=True);time.sleep(60)"
    client=subprocess.Popen([sys.executable,'-c',client_code,str(tmp_path/'jobs'),code,str(marker)],stdout=subprocess.PIPE,text=True)
    try:
        path=Path(client.stdout.readline().strip());deadline=time.monotonic()+5
        while not marker.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert marker.exists();record=wait_record(path,'running')
        assert execution.recover(path,trusted=True)['status']=='running'
        if kill_owner=='client':
            client.kill();client.wait(timeout=3);record=wait_record(path,'timed_out')
        else:
            # Keep a real process handle before killing it; no unowned PID-tree kill.
            from nil._execution_windows import _Runtime
            runtime=_Runtime();handle=runtime.OpenProcess(0x1|0x100000|0x1000,False,record['supervisor_pid'])
            assert handle
            try:runtime.terminate(handle);runtime.wait(handle,5)
            finally:runtime.close(handle)
            deadline=time.monotonic()+5
            while (execution._identity(int(marker.read_text())) is not None or execution._identity(record['pid']) is not None) and time.monotonic()<deadline:time.sleep(.02)
            record=execution.recover(path,trusted=True)
            assert record['status']=='interrupted'
        assert execution._identity(int(marker.read_text())) is None
    finally:
        if client.poll() is None:client.kill();client.wait(timeout=3)
        client.stdout.close()


def test_windows_recovery_does_not_signal_reused_pid_and_unsupported_limits_refuse(tmp_path):
    job=execution.submit('builtins:sum',[1,2],job_root=tmp_path/'jobs');assert job.result()==3
    from nil.lifecycle import atomic_json
    record=job.status();record.update(status='running',supervisor_identity='invalid',identity='invalid',pid=os.getpid())
    atomic_json(job.path/'job.json',record)
    assert execution.recover(job.path,trusted=True,result=True)==3
    with pytest.raises(NotImplementedError):execution.submit('builtins:sum',[1],limits=execution.Limits(memory_mb=128),job_root=tmp_path/'unsupported')
    assert not (tmp_path/'unsupported').exists()


def test_windows_native_struct_layout_and_handle_inheritance_policy():
    from nil._execution_windows import _Runtime
    runtime=_Runtime();C=runtime.C
    assert C.sizeof(runtime.Extended)==(144 if C.sizeof(C.c_void_p)==8 else 112)
    assert C.sizeof(runtime.Accounting)==48
    assert C.sizeof(runtime.StartupEx)==(112 if C.sizeof(C.c_void_p)==8 else 72)
    assert C.sizeof(runtime.Information)==(24 if C.sizeof(C.c_void_p)==8 else 16)
    handle=runtime.create_job()
    try:assert not os.get_handle_inheritable(handle)
    finally:runtime.close(handle)


def test_windows_assignment_failure_kills_real_suspended_process_before_user_code(tmp_path,monkeypatch):
    from nil import _execution_windows as windows
    marker=tmp_path/'must-not-run';created=[];original=windows._Runtime.create_suspended
    def create(runtime,*args):
        result=original(runtime,*args);created.append(result[2]);return result
    def refuse(*args):raise OSError('deliberate job assignment refusal')
    monkeypatch.setattr(windows._Runtime,'create_suspended',create)
    monkeypatch.setattr(windows._Runtime,'assign',refuse)
    with pytest.raises(OSError,match='assignment refusal'):
        windows.launch([sys.executable,'-c',"from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ran')",str(marker)])
    assert created and not marker.exists()
    assert execution._identity(created[0]) is None


def test_windows_json_publication_waits_for_actual_reader_lock(tmp_path):
    import threading
    from nil.lifecycle import atomic_json
    path=tmp_path/'journal.json';atomic_json(path,{'status':'running'})
    from nil._execution_windows import _Runtime
    runtime=_Runtime();handle=runtime.win.CreateFile(str(path),0x80000000,1,0,3,0x80,0);errors=[]
    def write():
        try:atomic_json(path,{'status':'completed'})
        except Exception as error:errors.append(error)
    thread=threading.Thread(target=write);thread.start()
    try:
        time.sleep(.05)
        assert thread.is_alive() and json.loads(path.read_text())['status']=='running'
    finally:runtime.close(handle)
    thread.join(timeout=3)
    assert not thread.is_alive() and not errors
    assert json.loads(path.read_text())['status']=='completed'
