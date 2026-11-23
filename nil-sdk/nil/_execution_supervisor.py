"""Detached stdlib-only job supervisor. No caller thread owns its lifetime."""
import json
import os
from pathlib import Path
import subprocess
import sys

from nil.execution import Limits, _Supervisor, _identity, _finish_runs
from nil.lifecycle import atomic_json


def main():
    path = Path(sys.argv[1])
    record = json.loads((path / 'job.json').read_text())
    record.update(supervisor_pid=os.getpid(), supervisor_identity=_identity(os.getpid()))
    atomic_json(path / 'job.json', record)
    worker = None
    quota = None
    tree = None
    try:
        limits=Limits(**record['limits'])
        if limits.aggregate_memory_mb is not None:
            from nil._execution_resources import Cgroup
            quota=Cgroup(limits.cgroup_root,path.name,limits.aggregate_memory_mb)
            record['cgroup_path']=str(quota.path)
            atomic_json(path/'job.json',record)
        args=[sys.executable,'-m','nil._execution_worker',str(path)]
        if os.name=='nt':
            from nil._execution_windows import launch
            worker,tree=launch(args)
            record['containment']='windows-job-object'
        else:
            worker = subprocess.Popen(args,start_new_session=True,stdin=subprocess.DEVNULL,
                                      stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            record['containment']='posix-process-group'
        record.update(status='running', pid=worker.pid, identity=_identity(worker.pid))
        atomic_json(path / 'job.json', record)
        _Supervisor(path, worker, limits, quota=quota,tree=tree)._watch()
    except BaseException as exc:
        if worker is not None:
            try:
                if tree is not None: tree.stop()
                else:
                    import signal
                    os.killpg(worker.pid,signal.SIGKILL);worker.wait()
            except (OSError,subprocess.TimeoutExpired): pass
        if quota is not None:
            try: quota.kill(); quota.close()
            except OSError: pass
        if tree is not None:
            try: tree.close()
            except OSError: pass
        error = f'Supervisor failed: {type(exc).__name__}: {exc}'
        record.update(status='failed', error=error)
        _finish_runs(path, 'failed', error)
        atomic_json(path / 'job.json', record)
        (path / 'input.pkl').unlink(missing_ok=True)
        (path / 'result.pkl').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
