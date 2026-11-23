# execution

Opt-in owned workers for trusted local operations: POSIX process groups or
Windows Job Objects. Importing NIL starts no worker/watchdog or native DLL calls.

## Usage

```python
import nil

job = nil.execution.submit('builtins:sum', [1, 2, 3],
    limits=nil.execution.Limits(timeout=30, threads=1), job_root='./nil_jobs')
print(job.result())  # 6
print(job.status())
```

## Parameters

`submit(operation, *args, limits=Limits(), job_root='./nil_jobs', **kwargs)` accepts
an importable callable or `module:function` string. `run` waits for the result.
Limits: timeout seconds, optional memory_mb, threads, log_bytes and payload_mb.
`job.cancel()` requests group termination. `job.result()` waits and raises typed
timeout/cancellation/execution errors. `recover(path, trusted=True, result=False)`
reconciles a dead Linux/Windows worker without signalling a saved PID; result=True loads
a completed trusted result. `capabilities()` reports the actual platform scope.

## Output

Job directory with durable job/outcome JSON, bounded combined log and integrity
checked result. Input serialization is removed when the watchdog finishes.
Classical/DL results contain a worker-owned model copy; the caller's model is
unchanged. Registered Run metadata is marked failed/cancelled on worker failure.
Linux reports worker peak RSS separately from its address-space limit.

## Requirements

Core Python for execution; the operation's optional backends must already exist
in the same interpreter. Execution needs POSIX or CPython on Windows. Linux memory controls remain
Linux-only; identity-safe recovery supports Linux and Windows. Standard pickle needs importable operations,
model classes and data types. Define custom functions/classes in importable files.

## Limitations

This is operational isolation for trusted code, not a security sandbox. Pickles
can execute code; never load another party's job directory as trusted. `memory_mb` is
a per-process virtual address-space cap, not aggregate RSS or GPU RAM; libraries
may require substantial virtual mappings even for small inputs. Native process
descendants inherit limits but can consume additional aggregate resources.
Thread environment settings are requests to native runtimes, not universal caps.
Logs and serialized input/results are bounded; arbitrary operation-owned files
are not. Timeout starts after input serialization. Native state/results are
copied across processes, which adds optional serialization/storage overhead.
A detached stdlib supervisor owns the deadline, cancellation marker, bounded log
and owned tree; it operates independently of client exit/termination. The deadline includes supervisor
and worker startup after input serialization. Killing the supervisor itself or
the machine is outside this guarantee. recover never signals a saved PID, including
when the supervisor has died. Windows kernel acceptance and macOS memory acceptance remain
open checklist items; Windows implementation is described below. Checkpoints retain
their own save/resume contracts. A timeout does not guarantee a resumable partial
epoch. Completed results and job records remain until explicitly removed.


## Process-group RSS guard (Linux)

`Limits(group_rss_mb=128)` opts into a detached-supervisor check of summed resident
bytes across the owned process group, approximately every 20 ms. Shared pages
count per process; sampled peaks may miss transient spikes. Budget breach kills
the owned group and raises `ExecutionResourceLimit`; durable job status is
`resource_limited`, with `group_memory` limit/peak/member evidence. The guard
survives submitting-client exit. Missing procfs permissions fail the requested
guarantee. It is a sampled operational guard, not a hard allocation quota; child
processes creating new sessions can escape this particular group guard.

## Kernel aggregate memory (Linux, explicitly delegated)

`Limits(aggregate_memory_mb=128, cgroup_root='/absolute/delegated/root')` requests
a new owned cgroup-v2 child with `memory.max`, `memory.swap.max=0` and
`memory.oom.group=1`. The raw worker attaches before input deserialization; its
children inherit membership even when creating new process sessions. Deadline
and completion cleanup use `cgroup.kill` before removing the owned hierarchy.
Kernel OOM reports `ExecutionResourceLimit` and memory event/peak evidence. This
limits kernel-accounted memory, not GPU memory or unique RSS.

The administrator must provide a writable delegated cgroup-v2 root with the
memory controller already enabled for children. NIL does not change system
delegation, enable root controllers or silently substitute sampled enforcement.
`ResourceUnavailable` rejects absent capabilities before launching user code.
`capabilities(cgroup_root=...)` reports preflight availability; kernel creation/
attachment is checked again at startup. Real OOM/new-session tests require
`NIL_TEST_CGROUP_ROOT`; they remain unaccepted on a nondelegated host.

Run `scripts/accept_execution.py --require-rss --output report.json` for actual
local checks; add `--cgroup-root ...` to require real kernel containment. Missing
required capabilities produce exit 2 and a failed report, never a skipped pass.
Cross-platform and GPU acceptance still follow PHASES.md.


## Windows owned process trees

CPython on Windows 10+/Server 2016+ uses a native Job Object with KILL_ON_JOB_CLOSE, no breakaway
permission, and only the detached supervisor holding its job handle. The worker
is assigned during creation with JOB_LIST, remains suspended until membership
is verified, and then resumes. Assignment/setup
failures terminate the suspended process; they never fall back to an unowned
worker. Inherited handles are restricted to NUL stdin and the bounded stdout/
stderr pipe. Ordinary child/grandchild process creation inherits membership,
including new process groups. Cleanup terminates the job on success, failure,
timeout or cancellation and waits for active job processes to reach zero. Closing
the supervisor's last handle is also configured to kill members on owner death.

Windows cancellation terminates processes; it does not run Python cleanup hooks.
Use committed checkpoints for recovery. Durable records identify
`containment='windows-job-object'`. Saved PID creation-time identities are used
for read-only recovery; recovery never kills a PID from a journal. Journal
publication retries transient Windows reader sharing locks for at most 0.5 s.

`capabilities()['owned_process_tree']` covers both backends; `process_groups`
remains POSIX-only. `--require-windows` on scripts/accept_execution.py refuses
non-Windows hosts, rather than treating portable mocks as native acceptance.
Actual Windows kernel tests are wired into the three Windows Python CI cells
and remain unrun on the Linux development host. Job assignment must be allowed
by any enclosing host job; failure is explicit. External services, scheduled
tasks and WMI-created processes are outside this CreateProcess-owned tree.
No Windows RAM/CPU/GPU quotas are claimed. See docs/WINDOWS_EXECUTION.md.
