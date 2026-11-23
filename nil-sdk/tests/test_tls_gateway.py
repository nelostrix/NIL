import concurrent.futures
import http.client
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import socket
import ssl
import threading
import time
import pytest
import nil
from nil._gateway import ServingError


@pytest.fixture
def proxy(tls_certificates):
    cert=tls_certificates;state={'calls':0,'reject':0,'sleep':0.,'auth':[]}
    class Backend(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):self.reply(200,{'status':'ready'})
        def reply(self,status,body):
            body=json.dumps(body).encode();self.send_response(status);self.send_header('Content-Length',str(len(body)));self.end_headers()
            try:self.wfile.write(body)
            except OSError:pass
        def do_POST(self):
            state['calls']+=1;state['auth'].append(self.headers.get('Authorization'))
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if state['sleep']:time.sleep(state['sleep'])
            if state['reject']:state['reject']-=1;self.reply(503,{'error':'busy'});return
            self.reply(200,{'outputs':[[value*2 for value in row] for row in payload['inputs']]})
    backend=ThreadingHTTPServer(('127.0.0.1',0),Backend)
    thread=threading.Thread(target=backend.serve_forever,daemon=True);thread.start()
    server=nil.serving.create_gateway(upstream_port=backend.server_port,backend_token='private-backend-capability',token='client-capability-secret',
        certfile=cert/'server.crt',keyfile=cert/'server.key',client_ca=cert/'ca.crt',socket_timeout=.25,upstream_timeout=.5,request_slots=4)
    gateway_thread=threading.Thread(target=server.serve_forever,daemon=True);gateway_thread.start()
    def client(**options):
        return nil.serving.client(f'https://localhost:{server.server_port}',options.pop('token','client-capability-secret'),cafile=cert/'ca.crt',certfile=cert/'client.crt',keyfile=cert/'client.key',**options)
    def context(client_certificate=True):
        ctx=ssl.create_default_context(cafile=cert/'ca.crt')
        if client_certificate:ctx.load_cert_chain(cert/'client.crt',cert/'client.key')
        return ctx
    try:yield server,backend,state,client,context
    finally:
        server.shutdown();server.server_close();gateway_thread.join(timeout=3)
        backend.shutdown();backend.server_close();thread.join(timeout=3)


def test_actual_mtls_auth_private_backend_token_and_verified_client(proxy):
    server,backend,state,client,context=proxy
    assert client().predict([[1.,2.]])==[[2.,4.]]
    assert state['auth']==['Bearer private-backend-capability']
    with pytest.raises(ServingError) as error:client(token='wrong-token-capability').predict([[1]])
    assert error.value.status==401 and error.value.attempts==1 and state['calls']==1
    denied=nil.serving.client(f'https://localhost:{server.server_port}','client-capability-secret',cafile=None)
    with pytest.raises(ServingError,match='TLS certificate verification failed'):denied.predict([[1]])
    # A trusted server certificate alone does not satisfy client-certificate authentication.
    connection=http.client.HTTPSConnection('localhost',server.server_port,context=context(False),timeout=2)
    try:
        with pytest.raises((ssl.SSLError,OSError,http.client.HTTPException)):
            connection.request('POST','/predict',body=b'{}',headers={'Authorization':'Bearer client-capability-secret'});connection.getresponse()
    finally:connection.close()
    assert state['calls']==1


def test_retries_are_bounded_and_auth_is_never_retried(proxy):
    server,backend,state,client,context=proxy
    state['reject']=2
    assert client(max_attempts=3,retry_delay=.01).predict([[3]])==[[6]] and state['calls']==3
    state['reject']=10
    with pytest.raises(ServingError) as error:client(max_attempts=2,retry_delay=.01).predict([[3]])
    assert error.value.attempts==2 and state['calls']==5
    assert 'client-capability-secret' not in str(error.value)


@pytest.mark.parametrize('framing',[
    'Content-Length: 2\r\nContent-Length: 2\r\n',
    'Transfer-Encoding: chunked\r\nContent-Length: 2\r\n',
    'Authorization: Bearer other-capability-token\r\nContent-Length: 2\r\n',
])
def test_ambiguous_headers_do_not_reach_backend(proxy,framing):
    server,backend,state,client,context=proxy
    with socket.create_connection(('127.0.0.1',server.server_port),timeout=2) as raw:
        with context().wrap_socket(raw,server_hostname='localhost') as secured:
            secured.sendall(('POST /predict HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer client-capability-secret\r\n'+framing+'\r\n{}').encode())
            assert b' 400 ' in secured.recv(4096)
    assert state['calls']==0


def test_dripping_body_cannot_extend_the_absolute_read_deadline(proxy):
    server,backend,state,client,context=proxy
    started=time.monotonic()
    with socket.create_connection(('127.0.0.1',server.server_port),timeout=2) as raw:
        with context().wrap_socket(raw,server_hostname='localhost') as secured:
            secured.sendall(b'POST /predict HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer client-capability-secret\r\nContent-Type: application/json\r\nContent-Length: 1000\r\n\r\n')
            for _ in range(20):
                try:secured.sendall(b' ');time.sleep(.04)
                except OSError:break
            assert time.monotonic()-started<.8
    deadline=time.monotonic()+1
    while server.metrics()['active_connections'] and time.monotonic()<deadline:time.sleep(.01)
    assert server.metrics()['active_connections']==0 and server.metrics()['connection_timeouts']>=1
    assert state['calls']==0 and client().predict([[2]])==[[4]]


def test_warm_tls_load_is_bounded_and_gateway_recovers(proxy):
    server,backend,state,client,context=proxy
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda value:client(max_attempts=5,retry_delay=.01).predict([[value]]),range(24)))
    assert results==[[[value*2]] for value in range(24)]
    assert server.metrics()['peak_connections']<=4
    state['sleep']=1
    started=time.monotonic()
    with pytest.raises(ServingError):client(max_attempts=1,timeout=.15).predict([[1]])
    assert time.monotonic()-started<.5


def test_header_budgets_and_expect_rejection_precede_backend_work(proxy):
    server,backend,state,client,context=proxy
    for extra,status in [('X-Large: '+'x'*5000+'\r\n',b' 431 '),('Expect: 100-continue\r\n',b' 417 ')]:
        with socket.create_connection(('127.0.0.1',server.server_port),timeout=2) as raw:
            with context().wrap_socket(raw,server_hostname='localhost') as secure:
                secure.sendall(('POST /predict HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer client-capability-secret\r\nContent-Length: 100\r\n'+extra+'\r\n').encode())
                assert status in secure.recv(4096)
    assert state['calls']==0


def test_stalled_tls_handshakes_are_bounded_and_release_admission(proxy):
    server,backend,state,client,context=proxy;sockets=[]
    try:
        for _ in range(4):sockets.append(socket.create_connection(('127.0.0.1',server.server_port),timeout=2))
        deadline=time.monotonic()+.15
        while server.metrics()['active_connections']<4 and time.monotonic()<deadline:time.sleep(.005)
        assert server.metrics()['active_connections']==4
        with pytest.raises(ServingError):client(max_attempts=1).predict([[1]])
        assert server.metrics()['admission_rejections']>=1
        time.sleep(.3)
        assert client().predict([[1]])==[[2]]
    finally:
        for connection in sockets:connection.close()
    assert server.metrics()['peak_connections']<=4


def test_public_client_rejects_invalid_origins_budgets_and_exposes_typed_error():
    assert nil.serving.ServingError is ServingError
    for url in ('http://localhost','https://user:password@localhost','https://localhost/path','https://localhost:0'):
        with pytest.raises(ValueError):nil.serving.client(url,'client-capability-secret')
    for options in ({'max_attempts':True},{'timeout':float('inf')},{'max_response_bytes':0}):
        with pytest.raises(ValueError):nil.serving.client('https://localhost','client-capability-secret',**options)


def test_metrics_require_auth_and_expose_only_aggregate_counters(proxy,capfd):
    server,backend,state,client,context=proxy
    def metrics(token):
        connection=http.client.HTTPSConnection('localhost',server.server_port,context=context(),timeout=2)
        try:
            connection.request('GET','/metrics',headers={'Authorization':'Bearer '+token})
            response=connection.getresponse();return response.status,json.loads(response.read())
        finally:connection.close()
    status,body=metrics('wrong-private-capability')
    assert status==401 and body=={'error':'Unauthorized'}
    assert client().predict([[12345.6789]])==[[24691.3578]]
    status,body=metrics('client-capability-secret')
    assert status==200 and body['auth_rejections']==1 and body['successes']==1
    assert body['latency_seconds']>0 and body['peak_connections']<=4
    assert all(type(value) in (int,float) for value in body.values())
    text=json.dumps(body)+''.join(capfd.readouterr())
    assert all(secret not in text for secret in ('client-capability-secret','private-backend-capability','12345.6789','24691.3578'))
    assert state['calls']==1  # Metrics are local; they never reach inference.
