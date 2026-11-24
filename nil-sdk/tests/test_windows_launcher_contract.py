"""Portable launch-order/failure contracts, not positive Windows acceptance."""
import io
import pytest
from nil import _execution_windows as windows


class FakeRuntime:
    def __init__(self,fail=None):self.events=[];self.fail=fail;self.log=io.BytesIO(b'owned output')
    def event(self,name,*args):
        self.events.append((name,*args))
        if self.fail==name:raise OSError('deliberate '+name)
    def create_job(self):self.event('job');return 1
    def create_pipe(self):self.event('pipe');return 2,3
    def create_input(self):self.event('input');return 4
    def create_suspended(self,args,write,stdin,job):self.event('suspended',args,write,stdin,job);return 5,6,7,8
    def assign(self,job,process):self.event('assign',job,process)
    def stdout(self,handle):self.event('stdout',handle);return self.log
    def resume(self,thread):self.event('resume',thread)
    def terminate(self,process):self.event('terminate',process)
    def wait(self,process,timeout):self.event('wait',process,timeout);return 0
    def close(self,handle):self.event('close',handle)
    def kill_job(self,handle):self.event('kill_job',handle)
    def active(self,handle):self.event('active',handle);return 0
    def poll(self,handle):self.event('poll',handle);return 0


def test_worker_is_assigned_before_resume_and_job_handle_is_retained(monkeypatch):
    runtime=FakeRuntime();monkeypatch.setattr(windows,'_Runtime',lambda:runtime)
    process,tree=windows.launch(['python','-m','nil._execution_worker','a path with spaces'])
    names=[event[0] for event in runtime.events]
    assert next(event for event in runtime.events if event[0]=='suspended')[-1]==1
    assert names.index('suspended')<names.index('assign')<names.index('resume')
    assert ('close',1) not in runtime.events and ('close',5) not in runtime.events
    assert all(('close',handle) in runtime.events for handle in (3,4,6))
    tree.stop();assert ('kill_job',1) in runtime.events
    assert process.wait()==0 and process.stdout.read()==b'owned output'
    tree.close();tree.close()
    assert runtime.events.count(('close',1))==1 and runtime.events.count(('close',5))==1


@pytest.mark.parametrize('failure',['pipe','input','suspended','assign','resume'])
def test_startup_failure_never_leaves_an_unowned_suspended_worker(monkeypatch,failure):
    runtime=FakeRuntime(failure);monkeypatch.setattr(windows,'_Runtime',lambda:runtime)
    with pytest.raises(OSError,match='deliberate '+failure):windows.launch(['python'])
    assert ('close',1) in runtime.events
    if failure in ('assign','resume'):
        assert ('terminate',5) in runtime.events and ('close',5) in runtime.events
        assert ('close',6) in runtime.events
    if failure=='assign':assert not any(event[0]=='resume' for event in runtime.events)
    if failure=='resume':assert runtime.log.closed


def test_required_windows_acceptance_refuses_linux_instead_of_passing_mocks(tmp_path):
    import json,os,subprocess,sys
    from pathlib import Path
    if os.name=='nt':pytest.skip('Negative-capability test is for non-Windows hosts')
    root=Path(__file__).resolve().parents[2];output=tmp_path/'windows.json'
    result=subprocess.run([sys.executable,str(root/'scripts/accept_execution.py'),'--require-windows','--output',str(output)],capture_output=True,text=True,timeout=10)
    assert result.returncode==2,result.stderr
    report=json.loads(output.read_text())
    assert report['status']=='failed' and not report['checks'] and 'actual Windows' in report['error']


def test_post_resume_setup_failure_still_closes_the_owned_job(monkeypatch):
    runtime=FakeRuntime();original=runtime.close;failed=[]
    def close(handle):
        if handle==6 and not failed:
            failed.append(handle);raise OSError('thread handle close failed')
        original(handle)
    runtime.close=close;monkeypatch.setattr(windows,'_Runtime',lambda:runtime)
    with pytest.raises(OSError,match='handle close failed'):windows.launch(['python'])
    assert ('terminate',5) in runtime.events and ('close',1) in runtime.events
    assert runtime.log.closed


def test_windows_sharing_retry_keeps_previous_record_until_atomic_publication(tmp_path,monkeypatch):
    import json
    from nil import lifecycle
    target=tmp_path/'record.json';source=tmp_path/'next.json'
    target.write_text('{"status":"running"}');source.write_text('{"status":"completed"}')
    original=lifecycle.os.replace;attempts=[]
    def replace(first,second):
        attempts.append(1)
        if len(attempts)<3:
            assert json.loads(target.read_text())['status']=='running'
            error=PermissionError('Windows reader sharing lock');error.winerror=32;raise error
        return original(first,second)
    monkeypatch.setattr(lifecycle.os,'replace',replace)
    lifecycle._replace_atomic(source,target,windows=True)
    assert len(attempts)==3 and json.loads(target.read_text())['status']=='completed'


def test_unrelated_windows_file_error_is_not_retried_or_published(tmp_path,monkeypatch):
    from nil import lifecycle
    target=tmp_path/'record';source=tmp_path/'next';target.write_text('old');source.write_text('new');attempts=[]
    def fail(*args):
        attempts.append(1);error=PermissionError('Unrelated failure');error.winerror=123;raise error
    monkeypatch.setattr(lifecycle.os,'replace',fail)
    with pytest.raises(PermissionError):lifecycle._replace_atomic(source,target,windows=True)
    assert len(attempts)==1 and target.read_text()=='old' and source.read_text()=='new'


@pytest.mark.parametrize('winerror',[5,32,33,None])
def test_status_read_retries_only_windows_sharing_race(tmp_path,monkeypatch,winerror):
    from nil import lifecycle
    import time
    path=tmp_path/'job.json';path.write_text('{"status":"completed"}')
    original=type(path).read_text;calls=[]
    def transient(self,**kwargs):
        calls.append(1)
        if len(calls)<3:
            error=PermissionError(13,'Windows atomic rename sharing race')
            if winerror is not None:error.winerror=winerror
            raise error
        return original(self,**kwargs)
    monkeypatch.setattr(type(path),'read_text',transient)
    monkeypatch.setattr(time,'sleep',lambda _:None)
    assert lifecycle._read_json(path,windows=True)=={'status':'completed'}
    assert len(calls)==3
    calls.clear()
    with pytest.raises(PermissionError):lifecycle._read_json(path,windows=False)
    assert len(calls)==1


def test_status_read_persistent_permission_and_corruption_fail_closed(tmp_path,monkeypatch):
    from nil import lifecycle
    import time
    path=tmp_path/'job.json';path.write_text('invalid JSON')
    with pytest.raises(ValueError):lifecycle._read_json(path,windows=True)
    def denied(self,**kwargs):
        error=PermissionError('persistent access denial');error.winerror=5;raise error
    monkeypatch.setattr(type(path),'read_text',denied)
    ticks=iter([0.,1.]);monkeypatch.setattr(time,'monotonic',lambda:next(ticks))
    with pytest.raises(PermissionError):lifecycle._read_json(path,windows=True)
