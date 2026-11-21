"""Real owned-process isolation; no fake native model results."""

import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

from nil import execution
from nil.lifecycle import atomic_json

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='POSIX process-group acceptance')


def test_owned_success_errors_and_trusted_recovery(tmp_path):
    job = execution.submit('builtins:sum', [1, 2, 3], job_root=tmp_path)
    assert job.result() == 6
    assert job.status()['status'] == 'completed'
    assert not (job.path / 'input.pkl').exists()
    with pytest.raises(ValueError, match='trusted=True'):
        execution.recover(job.path, result=True)
    assert execution.recover(job.path, trusted=True, result=True) == 6
    bad = execution.submit('builtins:int', 'wrong', job_root=tmp_path)
    with pytest.raises(execution.ExecutionError, match='ValueError'):
        bad.result()
    assert bad.status()['status'] == 'failed'
    assert not (bad.path / 'result.pkl').exists()


def test_timeout_and_cancellation_are_automatic(tmp_path):
    job = execution.submit('time:sleep', 30, limits=execution.Limits(timeout=.2), job_root=tmp_path)
    time.sleep(.8)  # Watchdog must operate even without result/status calls.
    assert job.status()['status'] == 'timed_out'
    with pytest.raises(execution.ExecutionTimeout):
        job.result()
    assert job.process.poll() is not None
    cancelled = execution.submit('time:sleep', 30, job_root=tmp_path)
    cancelled.cancel()
    with pytest.raises(execution.ExecutionCancelled):
        cancelled.result()
    assert cancelled.process.poll() is not None


def test_bounded_output_and_result_payload(tmp_path):
    job = execution.submit('builtins:print', 'x' * 200000,
                           limits=execution.Limits(log_bytes=1024), job_root=tmp_path)
    assert job.result() is None
    assert (job.path / 'worker.log').stat().st_size == 1024
    oversized = execution.submit('builtins:bytes', 2 * 1024 ** 2,
                                  limits=execution.Limits(payload_mb=1), job_root=tmp_path)
    with pytest.raises(execution.ExecutionError, match='payload_mb'):
        oversized.result()
    assert not (oversized.path / 'result.pkl').exists()


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux virtual address-space cap')
def test_address_space_cap_rejects_allocation_and_parent_survives(tmp_path):
    job = execution.submit('builtins:bytearray', 512 * 1024 ** 2,
                           limits=execution.Limits(memory_mb=192), job_root=tmp_path)
    with pytest.raises(execution.ExecutionError, match='MemoryError'):
        job.result()
    assert execution.run('builtins:sum', [3, 4], job_root=tmp_path) == 7


def test_child_group_cleanup_and_registered_run_state(tmp_path):
    # Native Python subprocess exercises owned descendants and real Run records.
    code = """
import subprocess, sys, time
from nil.lifecycle import Run
run = Run(sys.argv[1], 'worker-fixture')
run.status('running')
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
open(sys.argv[2], 'w').write(str(child.pid))
time.sleep(30)
"""
    child_file = tmp_path / 'child.pid'
    job = execution.submit('subprocess:run', [sys.executable, '-c', code, str(tmp_path / 'runs'), str(child_file)],
                           limits=execution.Limits(timeout=2), job_root=tmp_path / 'jobs', check=True)
    with pytest.raises(execution.ExecutionTimeout):
        job.result()
    assert child_file.exists()
    if sys.platform == 'linux':
        assert execution._identity(int(child_file.read_text())) is None
    (run_file,) = (tmp_path / 'runs').glob('*/run.json')
    record = json.loads(run_file.read_text())
    assert record['status'] == 'failed' and record['execution_status'] == 'timed_out'


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux PID identity recovery')
def test_recovery_never_signals_saved_pid_and_reconciles_finished_outcome(tmp_path, monkeypatch):
    job = execution.submit('builtins:sum', [1, 2], job_root=tmp_path)
    assert job.result() == 3
    record = job.status()
    record.update(status='running', pid=os.getpid(), identity='wrong-start-token')
    atomic_json(job.path / 'job.json', record)

    def forbidden(*args, **kwargs):
        raise AssertionError('Recovery must not signal any saved PID')

    monkeypatch.setattr(os, 'kill', forbidden)
    monkeypatch.setattr(os, 'killpg', forbidden)
    assert execution.recover(job.path, trusted=True, result=True) == 3
    (job.path / 'outcome.json').unlink()
    record['status'] = 'running'
    atomic_json(job.path / 'job.json', record)
    assert execution.recover(job.path, trusted=True)['status'] == 'interrupted'


def test_native_classical_worker_returns_model_copy_and_metrics(tmp_path):
    pytest.importorskip('sklearn')
    import nil
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted
    model = nil.model('linear_regression')
    result = execution.run('nil.classical:fit', model, [[0.], [1.], [2.]], [1., 3., 5.],
        validation=([[3.]], [7.]), output_dir=tmp_path / 'runs', job_root=tmp_path / 'jobs')
    assert result.metrics['mse'] == pytest.approx(0, abs=1e-20)
    assert result.predict([[4.]])[0] == pytest.approx(9)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_invalid_limits_and_unpicklable_inputs_do_not_launch_workers(tmp_path):
    for options in ({'timeout': 0}, {'threads': True}, {'memory_mb': -1}):
        with pytest.raises(ValueError):
            execution.Limits(**options)
    with pytest.raises((AttributeError, TypeError)):
        execution.submit(lambda: None, job_root=tmp_path)
    assert all(json.loads(path.read_text())['status'] == 'failed'
               for path in tmp_path.glob('*/job.json'))


def test_actual_dl_training_runs_in_owned_worker_and_writes_valid_checkpoint(tmp_path):
    torch = pytest.importorskip('torch')
    import nil
    torch.set_num_threads(1)
    model = nil.nn.mlp(1, 1, hidden=())
    data = nil.data.tensors(torch.arange(8, dtype=torch.float32).reshape(-1, 1),
                            torch.arange(8, dtype=torch.float32).reshape(-1, 1))
    result = execution.run('nil.training.trainer:train', model, data, epochs=2,
        device='cpu', output_dir=tmp_path / 'runs', job_root=tmp_path / 'jobs')
    assert result.status == 'completed' and len(result.history) == 2
    from nil.training.checkpoint import load
    restored = nil.nn.mlp(1, 1, hidden=())
    assert load(result.checkpoint, restored)['epoch'] == 2
    for first, second in zip(result.model.parameters(), restored.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux client-kill acceptance')
def test_detached_supervisor_enforces_deadline_after_client_sigkill(tmp_path):
    import subprocess
    client_code = '''
import sys, time
from nil import execution
job = execution.submit('subprocess:run', [sys.executable, '-c', sys.argv[3], sys.argv[2]],
    limits=execution.Limits(timeout=2), job_root=sys.argv[1])
print(job.path, flush=True)
time.sleep(60)
'''
    worker_code = '''
import subprocess, sys, time
from pathlib import Path
from nil.lifecycle import Run
run = Run(sys.argv[1], 'abandoned'); run.status('running')
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path(sys.argv[1], 'child.pid').write_text(str(child.pid))
time.sleep(60)
'''
    client = subprocess.Popen([sys.executable, '-c', client_code, str(tmp_path / 'jobs'),
                               str(tmp_path / 'runs'), worker_code], stdout=subprocess.PIPE, text=True)
    try:
        job_path = Path(client.stdout.readline().strip())
        deadline = time.monotonic() + 5
        child_file = tmp_path / 'runs' / 'child.pid'
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert child_file.exists(), 'Actual worker and descendant must start before killing client'
        client.kill(); client.wait(timeout=2)
        record = json.loads((job_path / 'job.json').read_text())
        assert record['supervisor_pid'] != client.pid
        assert execution.recover(job_path, trusted=True)['status'] == 'running'
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            record = json.loads((job_path / 'job.json').read_text())
            if record['status'] == 'timed_out' and not (job_path / 'input.pkl').exists():
                break
            time.sleep(.02)
        assert record['status'] == 'timed_out'
        assert execution._identity(record['pid']) is None
        assert execution._identity(int(child_file.read_text())) is None
        assert not (job_path / 'input.pkl').exists()
        runs = list((tmp_path / 'runs').glob('*/run.json'))
        assert runs and json.loads(runs[0].read_text())['status'] == 'failed'
    finally:
        if client.poll() is None: client.kill(); client.wait()
        client.stdout.close()


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux client-kill acceptance')
def test_result_is_recoverable_after_submitting_client_dies(tmp_path):
    import subprocess
    code = '''
import sys, time
from nil import execution
job = execution.submit('time:sleep', .4, job_root=sys.argv[1])
print(job.path, flush=True)
time.sleep(60)
'''
    client = subprocess.Popen([sys.executable, '-c', code, str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        job_path = Path(client.stdout.readline().strip())
        client.kill(); client.wait(timeout=2)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            record = json.loads((job_path / 'job.json').read_text())
            if record['status'] == 'completed' and not (job_path / 'input.pkl').exists(): break
            time.sleep(.02)
        assert execution.recover(job_path, trusted=True, result=True) is None
    finally:
        if client.poll() is None: client.kill(); client.wait()
        client.stdout.close()
