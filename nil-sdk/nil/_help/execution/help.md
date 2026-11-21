# execution

Opt-in owned POSIX workers for classical fitting, DL training and other trusted
local operations. Importing NIL or this module starts no worker/watchdog.

## Usage

```python
import nil

job = nil.execution.submit('nil.classical:fit',
    nil.model('ridge_regression'), [[0.], [1.], [2.]], [0., 2., 4.],
    limits=nil.execution.Limits(timeout=30, memory_mb=2048, threads=1),
    job_root='./nil_jobs', output_dir='./nil_runs')
result = job.result()
print(result.predict([[3.]]))
print(job.status())
```

## Parameters

`submit(operation, *args, limits=Limits(), job_root='./nil_jobs', **kwargs)` accepts
an importable callable or `module:function` string. `run` waits for the result.
Limits: timeout seconds, optional memory_mb, threads, log_bytes and payload_mb.
`job.cancel()` requests group termination. `job.result()` waits and raises typed
timeout/cancellation/execution errors. `recover(path, trusted=True, result=False)`
reconciles a dead Linux worker without signalling a saved PID; result=True loads
a completed trusted result. `capabilities()` reports the actual platform scope.

## Output

Job directory with durable job/outcome JSON, bounded combined log and integrity
checked result. Input serialization is removed when the watchdog finishes.
Classical/DL results contain a worker-owned model copy; the caller's model is
unchanged. Registered Run metadata is marked failed/cancelled on worker failure.
Linux reports worker peak RSS separately from its address-space limit.

## Requirements

Core Python for execution; the operation's optional backends must already exist
in the same interpreter. Owned groups require POSIX; memory caps and identity
recovery currently require Linux. Standard pickle needs importable operations,
model classes and data types. Define custom functions/classes in importable files.

## Limitations

This is operational isolation for trusted code, not a security sandbox. Pickles
can execute code; never load another party's job directory as trusted. Memory is
a per-process virtual address-space cap, not aggregate RSS or GPU RAM; libraries
may require substantial virtual mappings even for small inputs. Native process
descendants inherit limits but can consume additional aggregate resources.
Thread environment settings are requests to native runtimes, not universal caps.
Logs and serialized input/results are bounded; arbitrary operation-owned files
are not. Timeout starts after input serialization. Native state/results are
copied across processes, which adds optional serialization/storage overhead.
A detached stdlib supervisor owns the deadline, cancellation marker, bounded log
and worker group; it survives client exit/SIGKILL. The deadline includes supervisor
and worker startup after input serialization. Killing the supervisor itself or
the machine is outside this guarantee. recover never signals a saved PID, including
when the supervisor has died. Windows trees and macOS memory acceptance remain
open checklist items. Checkpoints retain
their own save/resume contracts. A timeout does not guarantee a resumable partial
epoch. Completed results and job records remain until explicitly removed.
