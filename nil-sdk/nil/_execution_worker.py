"""Trusted local operation worker. Invoked only by nil.execution."""

from importlib import import_module
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

from nil.lifecycle import atomic_json, sha256


def main():
    path = Path(sys.argv[1])
    record = json.loads((path / "job.json").read_text())
    limits = record["limits"]
    started = time.monotonic()
    try:
        if limits["memory_mb"] is not None:
            import resource
            cap = limits["memory_mb"] * 1024 ** 2
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        os.environ['_NIL_EXECUTION_DIRECTORY'] = str(path)
        with (path / "input.pkl").open('rb') as stream:
            operation, args, kwargs = pickle.load(stream)
        if isinstance(operation, str):
            module, symbol = operation.split(':', 1)
            operation = getattr(import_module(module), symbol)
        result = operation(*args, **kwargs)
        from nil.execution import _LimitedWriter
        artifact = path / "result.pkl"
        with artifact.open('wb') as stream:
            pickle.dump(result, _LimitedWriter(stream, limits["payload_mb"] * 1024 ** 2), protocol=5)
        outcome = {"status": "completed", "sha256": sha256(artifact)}
    except BaseException as exc:
        outcome = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:8000],
                   "traceback": traceback.format_exc(limit=12)[-12000:]}
    outcome['worker_elapsed'] = time.monotonic() - started
    if sys.platform == 'linux':
        import resource
        outcome['worker_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    atomic_json(path / 'outcome.json', outcome)


if __name__ == '__main__':
    main()
