"""Opt-in POSIX worker execution. Importing this module starts no process/thread."""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import threading
import time
import uuid

from nil.lifecycle import atomic_json, sha256

_TERMINAL = {"completed", "failed", "timed_out", "cancelled", "interrupted"}


@dataclass(frozen=True)
class Limits:
    timeout: float = 60.0
    memory_mb: int | None = None
    threads: int = 1
    log_bytes: int = 65536
    payload_mb: int = 256

    def __post_init__(self):
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        for name in ("threads", "log_bytes", "payload_mb", "memory_mb"):
            value = getattr(self, name)
            if name == "memory_mb" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class ExecutionError(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(f"Job {record['id']} {record['status']}: {record.get('error', '')}")


class ExecutionTimeout(ExecutionError, TimeoutError):
    pass


class ExecutionCancelled(ExecutionError):
    pass


def capabilities():
    return {"process_groups": os.name == "posix", "address_space_limit": sys.platform == "linux",
            "identity_safe_recovery": sys.platform == "linux", "aggregate_rss_limit": False,
            "gpu_memory_limit": False, "independent_supervisor": os.name == "posix"}


def _identity(pid):
    if sys.platform != "linux":
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


class _LimitedWriter:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.written = stream, limit, 0

    def write(self, data):
        if self.written + len(data) > self.limit:
            raise ValueError("Serialized payload exceeds payload_mb")
        self.written += len(data)
        return self.stream.write(data)


def _finish_runs(path, state, error):
    """Only update runs explicitly registered with this owned execution job."""
    if state == "completed":
        return
    for registered in (path / "runs").glob("*.json"):
        try:
            run = Path(json.loads(registered.read_text())["path"])
            record = json.loads((run / "run.json").read_text())
            if record.get("execution_job") == path.name and record.get("status") in {"created", "running"}:
                record.update(status="cancelled" if state == "cancelled" else "failed",
                              execution_status=state, error=error)
                atomic_json(run / "run.json", record)
        except (OSError, ValueError, KeyError, TypeError):
            continue


class _Supervisor:
    """Runs only inside the detached supervisor process."""
    def __init__(self, path, process, limits):
        self.path, self.process, self.limits = path, process, limits

    def _running(self):
        # Keep an exited group leader unreaped until its owned descendants have
        # been killed. Its PID cannot be reused while it remains our zombie.
        if hasattr(os, 'waitid') and hasattr(os, 'WNOWAIT'):
            return os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None
        return self.process.poll() is None

    def _stop(self):
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + .3
        while self._running() and time.monotonic() < deadline:
            time.sleep(.01)
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait()

    def _watch(self):
        record = json.loads((self.path / "job.json").read_text())
        started = record["started_monotonic"]

        def drain():
            remaining = self.limits.log_bytes
            with (self.path / "worker.log").open("wb") as stream:
                while chunk := self.process.stdout.read(4096):
                    if remaining:
                        stream.write(chunk[:remaining])
                        remaining -= min(remaining, len(chunk))

        reader = threading.Thread(target=drain, name=f"nil-log-{self.path.name}", daemon=True)
        reader.start()
        state, error = "failed", "Worker did not publish an outcome"
        try:
            while self._running():
                if (self.path / "cancel").exists():
                    state, error = "cancelled", "Cancelled by caller"
                    self._stop()
                    break
                if time.monotonic() - started >= self.limits.timeout:
                    state, error = "timed_out", "Wall-clock execution limit exceeded"
                    self._stop()
                    break
                time.sleep(.02)
            else:
                self._stop()
                outcome_file = self.path / "outcome.json"
                if outcome_file.is_file():
                    outcome = json.loads(outcome_file.read_text())
                    record.update(outcome)
                    state, error = outcome["status"], outcome.get("error", "")
                    if self.process.returncode != 0:
                        state, error = "failed", f"Worker exited with code {self.process.returncode}"
                else:
                    error = f"Worker exited with code {self.process.returncode} without an outcome"
        except BaseException as exc:
            state, error = "failed", f"Worker monitor failed: {type(exc).__name__}: {exc}"
            try:
                if self.process.returncode is None:
                    self._stop()
            except OSError:
                pass
        finally:
            reader.join(timeout=1)
            if not reader.is_alive():
                self.process.stdout.close()
            record.update(status=state, error=error, elapsed=time.monotonic() - started)
            if state != "completed":
                _finish_runs(self.path, state, error)
                (self.path / "result.pkl").unlink(missing_ok=True)
            atomic_json(self.path / "job.json", record)
            # Inputs may contain large or private data; the durable result/status
            # is sufficient after completion. Never retain duplicate inputs.
            (self.path / "input.pkl").unlink(missing_ok=True)

class Job:
    """Client handle; the independently running supervisor owns the deadline."""
    def __init__(self, path, process, limits):
        self.path, self.process, self.limits = path, process, limits

    def status(self):
        return json.loads((self.path / "job.json").read_text())

    def cancel(self):
        (self.path / "cancel").touch()

    def result(self):
        try:
            self.process.wait()
        except KeyboardInterrupt:
            self.cancel()
            self.process.wait()
            raise
        record = self.status()
        if record["status"] not in _TERMINAL:
            record = recover(self.path, trusted=True)
        if record["status"] == "timed_out":
            raise ExecutionTimeout(record)
        if record["status"] == "cancelled":
            raise ExecutionCancelled(record)
        if record["status"] != "completed":
            raise ExecutionError(record)
        return _read_result(self.path, record)


def _read_result(path, record):
    artifact = path / "result.pkl"
    if artifact.is_symlink() or artifact.stat().st_size > record["limits"]["payload_mb"] * 1024 ** 2 or sha256(artifact) != record.get("sha256"):
        raise ValueError("Job result integrity or size check failed")
    with artifact.open("rb") as stream:
        return pickle.load(stream)


def submit(operation, *args, limits=None, job_root="./nil_jobs", **kwargs):
    """Execute an importable callable or 'module:function' in an owned POSIX worker.

    Standard pickle is used for trusted local inputs/results. Closures and local
    classes are not supported. Results contain a worker-owned model copy.
    """
    limits = limits or Limits()
    if not isinstance(limits, Limits):
        raise TypeError("limits must be nil.execution.Limits")
    if os.name != "posix":
        raise NotImplementedError("Owned process-group execution currently requires POSIX")
    if limits.memory_mb is not None and sys.platform != "linux":
        raise NotImplementedError("Address-space limits are currently supported on Linux only")
    if not callable(operation) and not (isinstance(operation, str) and ':' in operation):
        raise TypeError("operation must be importable or a 'module:function' path")
    path = Path(job_root).resolve() / ("job-" + uuid.uuid4().hex)
    path.mkdir(mode=0o700, parents=True)
    record = {"schema_version": 1, "id": path.name, "status": "created", "limits": asdict(limits),
              "owner_pid": os.getpid(), "created_at": time.time()}
    atomic_json(path / "job.json", record)
    process = None
    try:
        with (path / "input.pkl").open("wb") as stream:
            pickle.dump((operation, args, kwargs), _LimitedWriter(stream, limits.payload_mb * 1024 ** 2), protocol=5)
        environment = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            environment[key] = str(limits.threads)
        record.update(started_monotonic=time.monotonic())
        atomic_json(path / "job.json", record)
        process = subprocess.Popen([sys.executable, '-m', 'nil._execution_supervisor', str(path)],
                                   env=environment, start_new_session=True, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Only the supervisor writes job.json after launch. This avoids racing a
        # fast completion with the client's initial running-status publication.
        return Job(path, process, limits)
    except BaseException as exc:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(path / "job.json", record)
        (path / "input.pkl").unlink(missing_ok=True)
        raise


def run(operation, *args, limits=None, job_root="./nil_jobs", **kwargs):
    return submit(operation, *args, limits=limits, job_root=job_root, **kwargs).result()


def recover(path, *, trusted=False, result=False):
    """Recover a dead worker's journal without signalling any saved PID.

    Live supervisors/workers are left running. A detached supervisor enforces the
    deadline even after the submitting client exits. Recovery never signals PIDs.
    """
    if not trusted:
        raise ValueError("Recovering local job records/results requires trusted=True")
    path = Path(path)
    if path.is_symlink() or (path / "job.json").is_symlink():
        raise ValueError("Job records must not use symlinks")
    record = json.loads((path / "job.json").read_text())
    if record.get("schema_version") != 1 or record.get("id") != path.name:
        raise ValueError("Unsupported job record")
    if record["status"] not in _TERMINAL:
        if sys.platform != "linux":
            raise NotImplementedError("Identity-safe active-job recovery requires Linux")
        supervisor = record.get("supervisor_identity")
        if supervisor is not None and _identity(record.get("supervisor_pid")) == supervisor:
            return record
        # A newly launched supervisor may not yet have recorded its identity.
        # Refuse recovery of that launch window rather than clobber its journal.
        if record["status"] == "created" and 0 <= time.monotonic() - record.get("started_monotonic", 0) < 5:
            return record
        alive = record.get("identity") is not None and _identity(record.get("pid")) == record["identity"]
        if not alive:
            outcome_file = path / "outcome.json"
            if outcome_file.is_file() and not outcome_file.is_symlink():
                outcome = json.loads(outcome_file.read_text())
                record.update(outcome)
                if record.get("status") == "completed":
                    artifact = path / "result.pkl"
                    if not artifact.is_file() or artifact.is_symlink() or artifact.stat().st_size > record["limits"]["payload_mb"] * 1024 ** 2 or sha256(artifact) != record.get("sha256"):
                        record.update(status="failed", error="Recovered result failed integrity validation")
            else:
                record.update(status="interrupted", error="Worker identity is no longer active")
            _finish_runs(path, record["status"], record.get("error", ""))
            atomic_json(path / "job.json", record)
            (path / "input.pkl").unlink(missing_ok=True)
    if result:
        if record["status"] != "completed":
            raise ExecutionError(record)
        return _read_result(path, record)
    return record
