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
    try:
        worker = subprocess.Popen([sys.executable, '-m', 'nil._execution_worker', str(path)],
                                  start_new_session=True, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        record.update(status='running', pid=worker.pid, identity=_identity(worker.pid))
        atomic_json(path / 'job.json', record)
        _Supervisor(path, worker, Limits(**record['limits']))._watch()
    except BaseException as exc:
        if worker is not None:
            try:
                import signal
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.wait()
        error = f'Supervisor failed: {type(exc).__name__}: {exc}'
        record.update(status='failed', error=error)
        _finish_runs(path, 'failed', error)
        atomic_json(path / 'job.json', record)
        (path / 'input.pkl').unlink(missing_ok=True)
        (path / 'result.pkl').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
