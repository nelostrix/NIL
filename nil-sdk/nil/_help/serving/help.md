# serving

Explicitly create a token-authenticated loopback development inference server.

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

CPU-only development service. No production TLS, multiuser isolation or batching. Sandbox/network policies may prohibit socket creation.
