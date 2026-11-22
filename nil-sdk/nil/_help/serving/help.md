# serving

Explicit development and managed TLS inference endpoints.

## Usage

```python
import nil
import secrets

model = nil.nn.mlp(1, 1, hidden=())
server = nil.serving.create_server(model, port=0, token=secrets.token_urlsafe(32))
# server.serve_forever()  # Explicitly start when ready; shutdown/close when finished.
server.server_close()
```

## Parameters

`model`, loopback port, token and maximum request-body bytes.

## Output

A local HTTP server object; construction reserves a socket.

## Requirements

torch and permission to bind a loopback socket.

## Limitations

`create_server` is CPU-only development serving. The opt-in managed TLS gateway
is described below; public deployment and multiuser isolation remain separate
acceptance gates. Network policies may prohibit socket creation.

## Isolated worker service

`create_isolated_server(model, token=..., port=0, request_slots=8,
inference_timeout=10)` returns an explicit loopback service with a lazy persistent
CPU subprocess. Run the entry point under `if __name__ == '__main__':` for spawn.
The model must be an importable trusted CPU nn.Module; tensor JSON requests alone
cross the HTTP interface. Every request needs `Authorization: Bearer <token>`.

`POST /predict` accepts `{"inputs": [[1.0]]}`. Authenticated `GET /health` and
`GET /metrics` expose readiness/admission and counts/latency/worker restarts.
Request threads, body/input/output/model sizes and inference wall time are
bounded. One inference at a time, no inference queue: competing inference gets
503. Deadline gives 504 and terminates the worker; crashes give 502. A subsequent
request starts a fresh worker from the frozen model. Model error details stay
private; invalid model input gives 422. Browser origins and non-loopback Hosts
are refused. Always call `shutdown()` and `server_close()` when finished.

This supports a supervised local inference service. Public deployment needs a
separate authenticated TLS gateway, process supervision and OS resource quotas.
It is not a hostile-model sandbox; tensor/body budgets are not aggregate RAM or
GPU quotas. See docs/CPU_COMPLETION.md for the deployment/fault acceptance scope.

## Explicit TLS gateway and client

`create_gateway(upstream_port=..., backend_token=..., token=..., certfile=...,
keyfile=..., client_ca=...)` creates a separate TLS gateway for the loopback
backend. Providing client_ca requires a valid client certificate as well as the
client Bearer token. Backend credentials are held by the gateway and replaced
on forwarding. No numerical backend is needed for the gateway/client themselves.

`client('https://localhost:8443', token, cafile=..., certfile=..., keyfile=...)`
returns a verified HTTPS inference client; `client.predict([[1.]])` returns outputs.
Catch `nil.serving.ServingError` for status/attempt information. Retries cover
429/502/503/504 and transient transport errors, bounded by max_attempts (<=5)
and timeout (network budget). Authentication errors are not retried. Use retries
only for stateless/idempotent prediction. OS DNS resolution has its own timeout;
network read/write/handshake deadlines after resolution are actively enforced.

The gateway bounds connection admission, TLS handshakes, headers, bodies,
upstream responses and absolute connection deadlines. Stalled connections release
slots; excess connections close before TLS rather than queueing indefinitely.
`GET /metrics` is authenticated and reports counts/durations without bodies/tokens.
All connections close after one request; ambiguous framing and duplicate identity
headers are rejected. `server_close()` waits for bounded handlers to finish.

`python -m nil.serving --config /etc/nil/service.json` runs the explicit lifecycle
using a trusted module:factory, protected secret files and optional mTLS. The
reference systemd unit adds restart/whole-unit cleanup and OS resource policies.
No service is installed, enabled or started by importing/installing NIL.

This is a managed single-tenant reference, built on Python http.server; it is not
an audited Internet ingress. Public deployment/edge review and broader workloads
remain acceptance gates. See docs/DEPLOYMENT.md for limits, measured local tests
and real kernel resource evidence.
