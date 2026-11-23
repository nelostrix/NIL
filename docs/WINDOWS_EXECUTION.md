# Windows owned execution — implementation and acceptance

The Windows backend is now implemented. Native Windows acceptance is pending:
this development host is Linux, has no Windows runtime and has not dispatched CI.
Portable launcher tests and Linux regressions cannot certify Windows behavior.

## Public use

```python
import nil
job = nil.execution.submit('builtins:sum', [1, 2, 3],
    limits=nil.execution.Limits(timeout=10), job_root='./jobs')
print(job.result())
```

No extra package is required. The backend uses ctypes plus CPython's standard
Windows `_winapi` module. Creation-time job assignment requires Windows 10+
or Windows Server 2016+. The configured matrix targets CPython 3.10–3.12.
`owned_process_tree` reports the backend; `process_groups` remains POSIX-specific.
Linux-only memory controls fail explicitly when requested on Windows.

## Ownership and lifetime

1. The client starts a detached supervisor with redirected standard handles.
2. The supervisor creates an unnamed, noninheritable Job Object configured with
   KILL_ON_JOB_CLOSE and no breakaway permission.
3. Native extended CreateProcessW creates the worker with JOB_LIST and
   CREATE_SUSPENDED, placing it in the job during creation. Its inherited handle
   list contains only NUL stdin and a stdout/stderr pipe, never the job handle.
4. Membership is verified before the initial worker thread resumes. This closes
   the separate create-then-assign orphan window described by Microsoft.
   On assignment/setup failure, the suspended worker is terminated and all owned
   handles close. There is no uncontained fallback.
5. Native process handles drive lifetime checks. The supervisor drains bounded
   logs, handles timeout/cancel, and terminates the entire job even when the main
   worker exits normally leaving children. Job accounting must reach zero active
   processes before normal cleanup finishes.
6. The sole job handle closes on supervisor exit/death, providing the configured
   kernel cleanup mechanism. Durable job records use Windows process creation-time
   tokens for identity-safe recovery. Recovery never signals a saved PID.

The design uses documented [Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)
and [job limit flags](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information),
with [creation-time assignment](https://devblogs.microsoft.com/oldnewthing/20230209-00/?p=107812)
through the extended process attribute list. Unsupported job-list attributes fail
before user execution; there is no older-Windows unowned fallback. It has no PowerShell/taskkill,
psutil, pywin32 or permanently running application service dependency.

## Acceptance commands and CI

On a real Windows interpreter with NIL installed:

```sh
python scripts/accept_execution.py --require-windows --output windows-execution.json
python -m pytest nil-sdk/tests/test_windows_execution.py nil-sdk/tests/test_windows_launcher_contract.py -q
```

The runner requires actual Windows; Linux returns exit 2 and a failed report.
`.github/workflows/sdk-platforms.yml` runs both commands in each Windows/Python
cell against the built installed wheel. It no longer treats an unsupported
Windows backend as a passing containment result. The stdlib acceptance runner
works without pytest; pytest is an explicit CI/development dependency only.

The native tests cover success/errors, Unicode/space paths, bounded logs/results,
children and grandchildren in new process groups, normal-exit/timeout/cancel
cleanup, client and supervisor death, suspended assignment refusal, native struct
layout/handle policy, read-only PID recovery and actual reader-sharing publication.
Portable tests separately verify launch order, resource cleanup on injected startup
failures and bounded Windows sharing-error retry. No native pass is recorded yet.

## Limits

This is trusted-code operational ownership, not a hostile-code security sandbox.
Ordinary CreateProcess descendants inherit membership; externally requested
services, scheduled tasks and WMI-created processes are outside that tree.
Nested/enclosing host jobs must permit assignment. Unsupported assignment fails
rather than requesting breakaway. Termination is immediate and does not promise
Python finally/atexit callbacks; recover from committed checkpoints.

Journal writes retry Windows sharing/lock errors for at most 0.5 seconds while
preserving the previous atomic record. Other errors propagate. No Windows
aggregate RAM/CPU or GPU quota is implemented by this change. Those remain
separate checklist tasks, as do macOS acceptance and the wider release matrix.


Final creation-time launch contracts/help/runner checks: **20 passed**. Installed
Linux execution/resource/portable-Windows contracts outside the source tree:
**26 passed, three explicit skips**. Installed stdlib execution acceptance passes
including successful/timeout child cleanup and the RSS guard. Fresh offline
installation adds only nil-sdk. Both development environments use the rebuilt
regular wheel; 197 Python sources match and 103 help guides are bundled.
Native Windows acceptance is still unrun; ten real-kernel cases and three Windows
CI cells require a target host. The Linux --require-windows report fails with
exit 2, so portable test success is not mistaken for Windows certification.
