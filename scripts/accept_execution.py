"""Real, dependency-free execution acceptance. Missing required capabilities fail.

PYTHONPATH=nil-sdk python scripts/accept_execution.py --require-rss --output report.json
A delegated host can additionally specify --cgroup-root /absolute/delegated/path.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time


def run(*,require_rss=False,cgroup_root=None,require_windows=False):
    from nil import execution
    capabilities=execution.capabilities(cgroup_root=cgroup_root)
    report={'schema_version':1,'platform':platform.platform(),'python':platform.python_version(),
        'capabilities':capabilities,'required':{'sampled_group_rss':require_rss,'cgroup_root':cgroup_root,'windows_job_objects':require_windows},
        'checks':[],'status':'failed'}
    missing=[]
    if not capabilities['owned_process_tree']: missing.append('owned process-tree backend')
    if require_windows and not capabilities['windows_job_objects']:missing.append('actual Windows Job Object backend')
    if require_rss and not capabilities['sampled_group_rss']: missing.append('Linux sampled process-group RSS')
    if cgroup_root is not None and not capabilities['kernel_aggregate_memory']['available']:
        missing.append(capabilities['kernel_aggregate_memory']['reason'])
    if missing: report.update(error='Required capability unavailable: '+', '.join(missing)); return report,2
    def check(name,function):
        started=time.monotonic()
        try: evidence=function()
        except Exception as error:
            failure={'name':name,'status':'failed','error':f'{type(error).__name__}: {error}'}
            if hasattr(error,'record'):failure['job']=error.record
            report['checks'].append(failure);return False
        report['checks'].append({'name':name,'status':'passed','seconds':time.monotonic()-started,'evidence':evidence}); return True
    with tempfile.TemporaryDirectory(prefix='nil-execution-accept-') as directory:
        root=Path(directory)
        def success():
            job=execution.submit('builtins:sum',[1,2,3],job_root=root/'success')
            assert job.result()==6; return {'job_status':job.status()['status']}
        def deadline():
            job=execution.submit('time:sleep',20,limits=execution.Limits(timeout=.5),job_root=root/'timeout')
            try: job.result()
            except execution.ExecutionTimeout: return {'job_status':job.status()['status']}
            raise AssertionError('Deadline did not terminate worker')
        def cancel():
            job=execution.submit('time:sleep',20,job_root=root/'cancel'); job.cancel()
            try: job.result()
            except execution.ExecutionCancelled: return {'job_status':job.status()['status']}
            raise AssertionError('Cancellation did not terminate worker')
        def descendant_cleanup():
            marker=root/'child.pid'
            code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); open(sys.argv[1],'w').write(str(p.pid)); time.sleep(20)"
            job=execution.submit('subprocess:run',[sys.executable,'-c',code,str(marker)],check=True,
                limits=execution.Limits(timeout=2),job_root=root/'descendants')
            try: job.result()
            except execution.ExecutionTimeout: pass
            else: raise AssertionError('Descendant workload did not time out')
            pid=int(marker.read_text())
            if sys.platform=='linux': assert execution._identity(pid) is None
            elif os.name=='nt':
                assert execution._identity(pid) is None
            else:
                probe=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True,timeout=3)
                assert not probe.stdout.strip() or probe.stdout.strip().startswith('Z')
            return {'job_status':job.status()['status'],'child_alive':False}
        def successful_descendant_cleanup():
            marker=root/'completed-child.pid'
            code="import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); open(sys.argv[1],'w').write(str(p.pid))"
            job=execution.submit('subprocess:run',[sys.executable,'-c',code,str(marker)],check=True,job_root=root/'completed-tree')
            job.result();pid=int(marker.read_text())
            if sys.platform=='linux' or os.name=='nt':assert execution._identity(pid) is None
            else:
                probe=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True,timeout=3)
                assert not probe.stdout.strip() or probe.stdout.strip().startswith('Z')
            return {'job_status':job.status()['status'],'child_alive':False}
        def rss():
            job=execution.submit('subprocess:run',[sys.executable,'-c','import time; x=bytearray(80*1024**2); time.sleep(20)'],check=True,
                limits=execution.Limits(timeout=10,group_rss_mb=64),job_root=root/'rss')
            try: job.result()
            except execution.ExecutionResourceLimit: return job.status()['group_memory']
            raise AssertionError('RSS workload did not exceed resource budget')
        def cgroup():
            code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; x=bytearray(80*1024**2); time.sleep(20)'],start_new_session=True); x=bytearray(80*1024**2); time.sleep(20)"
            job=execution.submit('subprocess:run',[sys.executable,'-c',code],check=True,
                limits=execution.Limits(timeout=15,aggregate_memory_mb=128,cgroup_root=cgroup_root),job_root=root/'cgroup')
            try: job.result()
            except execution.ExecutionResourceLimit:
                record=job.status(); assert not Path(record['cgroup_path']).exists(); return record['aggregate_memory']
            raise AssertionError('Kernel aggregate workload did not exceed memory budget')
        for name,function in [('success',success),('timeout',deadline),('cancellation',cancel),('descendant_cleanup',descendant_cleanup),('successful_descendant_cleanup',successful_descendant_cleanup)]:
            if not check(name,function): return report,1
        if require_rss and not check('sampled_group_rss',rss): return report,1
        if cgroup_root is not None and not check('kernel_aggregate_memory',cgroup): return report,1
    report['status']='passed'; return report,0


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--require-rss',action='store_true')
    parser.add_argument('--require-windows',action='store_true');parser.add_argument('--cgroup-root'); parser.add_argument('--output',type=Path)
    args=parser.parse_args(); report,code=run(require_rss=args.require_rss,cgroup_root=args.cgroup_root,require_windows=args.require_windows)
    text=json.dumps(report,indent=2)+'\n'
    if args.output: args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(text)
    print(text); return code

if __name__=='__main__': raise SystemExit(main())
