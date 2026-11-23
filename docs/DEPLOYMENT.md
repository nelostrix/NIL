# Managed inference reference and acceptance

Target: one Linux host, one trusted CPU FP32 Torch model, single operator/team,
loopback or managed private mTLS boundary. Model loading is explicit through an
operator-owned Python factory. Base NIL installation remains dependency-free;
Torch is an explicit optional backend. The TLS gateway and HTTPS client use only
standard-library modules, and start no network/process activity on import.

This reference is not an audited public Internet ingress. Python's
[http.server documentation](https://docs.python.org/3/library/http.server.html)
warns about production use. External ingress, authorization policy, revocation,
security review and real workload SLOs remain deployment-specific acceptance.

## Running the explicit service

1. Provision an interpreter with NIL and the chosen model's explicit backend.
   Provide an importable `my_model:build` factory returning a trusted CPU nn.Module.
2. Use operator-managed server certificates and client trust roots. Protect private
   keys and credential files. POSIX token files must have no group/other access.
   NIL does not create permanent CA trust, fetch models or install a service.
3. Adapt `deploy/service.example.json`: certificate/credential paths, allowed Host
   names, ports and bounded timeouts. Its default gateway is loopback port8443;
   the backend is loopback9001. Separate client/backend tokens are never supplied
   in command-line arguments or forwarded unchanged.
4. Start explicitly: `python -m nil.serving --config /etc/nil/service.json`.
   Readiness is published only after the frozen model worker loads. SIGTERM/INT
   stop ingress, finish bounded gateway handlers, then stop inference and its
   worker. Optional `--ready-file` records ports/PID without credentials.
5. `deploy/nil-inference.service` is an operator-installed systemd template with
   restart, KillMode=control-group, MemoryMax=1G, no swap, CPUQuota=100%, TasksMax=128,
   restricted filesystem access and no new privileges. Provision its nil user,
   app/venv and credential paths first. It is not installed/enabled by this SDK.

The template's `/opt/nil/venv/bin/python` is deliberately explicit; dry verification
on the development host reports that this deployment path is not provisioned.
The actual local resource acceptance below instead uses isolated transient user
units and the existing explicit interpreter. No permanent service was installed.

## Gateway and client contracts

TLS minimum1.2, server certificate validation and optional required client
certificates use Python [SSLContext](https://docs.python.org/3/library/ssl.html).
Client certificates plus the Bearer capability authenticate a single tenant;
this is not per-user RBAC. Client credentials never enter the backend request.

Connections/threads, TLS handshake/header/body time, aggregate connection lifetime,
header/body/response bytes and upstream time are bounded. Absolute read deadlines
prevent a body trickle from renewing its budget. Duplicate identity/length headers,
transfer encoding and Expect requests are refused. The upstream is a fixed loopback
port and accepts only declared health/predict routes; request forwarding is not an
arbitrary URL proxy. Request contents/tokens and trusted-model errors are not logged.

The client verifies CA/hostnames, accepts only HTTPS origins, bounds payload/response
sizes and retries only transient statuses/transports with bounded jitter/backoff.
Authentication failures are not retried. Retries require stateless/idempotent
prediction; use max_attempts=1 for other models. Its active network deadline covers
handshake/read/write after address resolution. The OS resolver can outlast that
budget; resolved requests are refused if their remaining budget is exhausted.

Health means the service can accept work, not a guarantee for every input shape.
Authenticated JSON metrics expose admissions, TLS/auth failures, successful work,
timeouts, worker restarts and accumulated latency without body/identity logging.

## Local acceptance and scope

Real tests cover server/client certificates, missing/untrusted certificates,
separate backend tokens, bounded retries, duplicate/framing/large headers,
stalled TLS and dripping-body deadlines, concurrent admission/load, actual Torch
worker inference/restart, protected secret files and CLI signal shutdown.
OpenSSL CLI is explicit test tooling for temporary certificates only; certificates
and tokens are removed, not included in the package or installed in a trust store.

`python scripts/accept_serving.py --output report.json` measures 64 warm requests
at concurrency4 for a 1x1 affine model, checking all predictions independently.
The declared warm end-to-end target is zero failed requests and observed p95<=250ms.
Warmup is recorded separately. This small workload does not certify arbitrary model
throughput, distribution tails or sustained long-duration Internet workloads.

The same gate passed inside a real transient user systemd unit with a 1GiB memory
ceiling, swap disabled, CPUQuota=100% and TasksMax=128:

```sh
systemd-run --user --wait --pipe --collect --unit=nil-serving-accept-local \
  -p MemoryMax=1G -p MemorySwapMax=0 -p CPUQuota=100% -p TasksMax=128 \
  /absolute/venv/bin/python /absolute/NIL/scripts/accept_serving.py \
  --require-unit-policy --output /absolute/report.json
```

Installed-wheel users can run the extracted bundle command directly. Source-tree
runs also need `--setenv=PYTHONPATH=/absolute/NIL/nil-sdk`. The unit verifies actual
kernel policy values before model load, not just configuration text. Resource
settings follow [systemd resource-control](https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html).

## Real NIL kernel containment

A fresh delegated unit can also run the original hard-memory and escaped-session
cleanup gates, which do not require changing ancestor controllers:

```sh
systemd-run --user --wait --pipe --collect --unit=nil-cgroup-accept-local \
  -p Delegate=memory -p MemoryMax=256M -p MemorySwapMax=0 -p CPUQuota=100% \
  /absolute/venv/bin/python /absolute/NIL/scripts/accept_cgroup.py \
  --output /absolute/kernel-report.json
```

The runner refuses arbitrary/current non-test groups. It moves only its own
controller PID within its new delegated unit, enables memory only there, then runs
real NIL cgroup tests. The suite passed all eight tests, including a kernel128MiB
limit with three OOM-killed members and new-session descendant cleanup. Peak memory
for the tested NIL job reached exactly134217728 bytes. Cleanup restores the owned
controller subtree and systemd collects the transient unit.

Reports: `docs/benchmarks/2026-11-22-cgroup-kernel*.json` and
`2026-11-22-tls-serving*.json`. These replace the earlier untested-delegation
status for this supported user-unit setup. Interactive sessions without delegation
still fail explicitly; GPU memory quotas remain unimplemented.

## Observability and credential acceptance

The lifecycle accepts optional `--metrics-file PATH --metrics-interval 5`.
It atomically writes a schema-versioned JSON snapshot containing uptime, PID,
status and aggregate gateway/backend counters, then writes `stopped` after
shutdown drains handlers and closes the worker. With no metrics file there is no
periodic work; enabling snapshots adds no monitoring package or background thread.
The systemd example uses RuntimeDirectory=nil, mode0700 and `/run/nil/metrics.json`.
Do not place snapshots in a web-served directory. Snapshot errors fail the service
rather than silently publishing stale healthy status; process supervision should
monitor service exit as well as snapshot age. Readiness and metrics paths must
be distinct from each other and configuration/credential/certificate inputs.

POSIX credential reads validate the opened descriptor: regular file, no
permissions for group/other users, no final symlink and no FIFO blocking. Tokens
are bounded, printable and validated before model loading; client and backend
capabilities must differ. The operator owns credential directories and ACLs on
Windows. Rotation/revocation requires controlled restart with replacement secrets;
there is no automatic reload or per-user identity audit.

Metrics routes require the same client authentication as inference. Tests verify
that unauthorized metrics requests fail, counters reflect real requests, secrets
and sample values are absent from metrics/stdout/stderr, invalid configuration
errors are sanitized, periodic snapshots advance and final active counts are zero.

| Deployment target | Acceptance scope |
| --- | --- |
| Linux, Python3.12, installed CPU Torch | Signal lifecycle, worker failure/restart, mTLS, protected credentials, snapshots, bounded ingress |
| Linux transient systemd unit | Actual resource-policy/load evidence above; not a permanent service installation |
| Windows/macOS | Base owned execution passed; managed service template and signal lifecycle are not certified |
| Public Internet or multi-tenant ingress | Requires deployment-specific review and workload acceptance |

The minimum/current Linux CPU backend workflow reruns the service tests against
the installed wheel. See [backend compatibility](BACKEND_COMPATIBILITY.md) for
exact profiles and their evidence; no skipped case counts as acceptance.
