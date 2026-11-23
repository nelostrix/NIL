"""Small loopback-only JSON inference server for development, not public hosting."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


def create_server(model, port=0, token=None, max_body_bytes=1024 * 1024):
    """Construct a server; caller explicitly starts serve_forever.

    Only CPU tensor input is supported. User-provided model code runs in this
    process. This is not a sandbox or a production serving scheduler.
    """
    from nil.inference import predict
    from nil.runtime import require

    torch = require("torch", "torch")
    if not token or len(token) < 16:
        raise ValueError("Provide a local capability token of at least 16 characters")
    if not isinstance(max_body_bytes, int) or max_body_bytes < 1:
        raise ValueError("max_body_bytes must be positive")
    model.cpu().eval()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # No request bodies, credentials or predictions in access logs.

        def send(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def guard(self):
            import secrets

            host = self.headers.get("Host", "").split(":")[0]
            if host not in {"localhost", "127.0.0.1"} or self.headers.get("Origin"):
                self.send(403, {"error": "Loopback non-browser clients only"})
                return False
            if not secrets.compare_digest(
                self.headers.get("Authorization", "").encode("utf-8"),
                f"Bearer {token}".encode("utf-8"),
            ):
                self.send(401, {"error": "Invalid capability token"})
                return False
            return True

        def do_GET(self):
            if not self.guard():
                return
            self.send(
                200 if self.path == "/health" else 404,
                {"status": "ready"} if self.path == "/health" else {"error": "Not found"},
            )

        def do_POST(self):
            if not self.guard():
                return
            if self.path != "/predict":
                self.send(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= max_body_bytes:
                    self.send(413, {"error": "Invalid or oversized request body"})
                    return
                payload = json.loads(self.rfile.read(length))
                inputs = torch.tensor(payload["inputs"], dtype=torch.float32)
                if inputs.ndim == 0 or not torch.isfinite(inputs).all():
                    raise ValueError("Input must be a finite tensor with a batch dimension")
                output = predict(model, inputs, device_name="cpu")
                if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all():
                    raise ValueError("Model must return a finite tensor")
                self.send(200, {"outputs": output.cpu().tolist()})
            except (ValueError, TypeError, KeyError, RuntimeError):
                self.send(422, {"error": "Input or output does not match the model contract"})

    class Server(HTTPServer):
        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(10)
            return connection, address

    return Server(("127.0.0.1", port), Handler)


def create_isolated_server(model,*,port=0,token,request_slots=8,inference_timeout=10,
                           startup_timeout=30,max_body_bytes=1048576,max_input_elements=262144,
                           max_output_elements=262144,max_model_bytes=134217728):
    """Opt-in bounded loopback serving with a restartable CPU worker process.

    Model construction is trusted; request bodies contain data only. Public
    ingress/TLS/tenant authentication belong to a separately configured gateway.
    """
    from nil._serving import isolated_server
    return isolated_server(model,port=port,token=token,request_slots=request_slots,
        inference_timeout=inference_timeout,startup_timeout=startup_timeout,
        max_body_bytes=max_body_bytes,max_input_elements=max_input_elements,
        max_output_elements=max_output_elements,max_model_bytes=max_model_bytes)


def create_gateway(**options):
    """Explicit bounded TLS/mTLS gateway; requires no numerical backend."""
    from nil._gateway import gateway
    return gateway(**options)


def client(url,token,**options):
    """HTTPS inference client with certificate verification and bounded retries."""
    from nil._gateway import Client
    return Client(url,token,**options)


def __getattr__(name):
    if name=='ServingError':
        from nil._gateway import ServingError
        return ServingError
    raise AttributeError(name)


if __name__=='__main__':
    from nil._service import main
    raise SystemExit(main())
