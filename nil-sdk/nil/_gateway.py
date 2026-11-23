"""Explicit TLS proxy and bounded inference client. Standard library only."""
import http.client
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
import math
import secrets
import socket
from socketserver import ThreadingMixIn
import ssl
import threading
import time
from urllib.parse import urlsplit


def _integer(value,name,maximum):
    if type(value) is not int or not 1<=value<=maximum:raise ValueError(f'{name} must be an integer in [1,{maximum}]')

def _seconds(value,name):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=300:raise ValueError(f'{name} must be finite, positive and <=300 seconds')

def _token(value):
    if not isinstance(value,str) or not 16<=len(value)<=4000 or any(not 33<=ord(c)<=126 for c in value):raise ValueError('Provide a printable token of 16-4000 characters')


def _abort(sock):
    try:socket.socket.shutdown(sock,socket.SHUT_RDWR)
    except OSError:pass


def _cancel(sock):
    timer=getattr(sock,'_nil_timer',None)
    if timer is not None:
        timer.cancel()
        if timer is not threading.current_thread():timer.join()
        sock._nil_timer=None


class _Headers:
    def __init__(self,stream,budget):self.stream,self.budget=stream,budget
    def readline(self,size=-1):
        line=self.stream.readline(min(4097,size) if size>=0 else 4097);self.budget-=len(line)
        if len(line)>4096 or self.budget<0:raise http.client.HTTPException('Header budget exceeded')
        return line


def gateway(*,upstream_port,backend_token,token,certfile,keyfile,client_ca=None,
            host='127.0.0.1',port=0,allowed_hosts=('localhost','127.0.0.1'),request_slots=16,
            socket_timeout=2,upstream_timeout=12,max_body_bytes=1048576,max_response_bytes=4194304):
    _integer(upstream_port,'upstream_port',65535);_integer(request_slots,'request_slots',64)
    _integer(max_body_bytes,'max_body_bytes',16777216);_integer(max_response_bytes,'max_response_bytes',16777216)
    _seconds(socket_timeout,'socket_timeout');_seconds(upstream_timeout,'upstream_timeout');_token(token);_token(backend_token)
    if type(port) is not int or not 0<=port<=65535:raise ValueError('Invalid gateway port')
    if not isinstance(host,str) or not host:raise ValueError('Explicit bind host required')
    if not isinstance(allowed_hosts,(tuple,list)) or not allowed_hosts or any(not isinstance(name,str) or not name or any(c in name for c in '\r\n/@:') for name in allowed_hosts):raise ValueError('Provide explicit allowed DNS/IPv4 host names')
    allowed_hosts=tuple(name.lower() for name in allowed_hosts)
    context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.minimum_version=ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile,keyfile)
    if client_ca:context.load_verify_locations(cafile=client_ca);context.verify_mode=ssl.CERT_REQUIRED
    counts={'connections':0,'admission_rejections':0,'tls_failures':0,'requests':0,'auth_rejections':0,
            'invalid_requests':0,'successes':0,'upstream_errors':0,'upstream_timeouts':0,'disconnects':0,
            'connection_timeouts':0,'active_connections':0,'peak_connections':0,'latency_seconds':0.0}
    lock=threading.Lock()
    def add(name,value=1):
        with lock:counts[name]+=value
    def snapshot():
        with lock:return dict(counts)
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def send(self,status,body):
            self.server.arm(self.connection,socket_timeout)
            if not isinstance(body,bytes):body=json.dumps(body,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.send_header('Connection','close');self.send_header('Cache-Control','no-store')
            if status in (429,503):self.send_header('Retry-After','1')
            self.end_headers();self.close_connection=True
            try:self.wfile.write(body)
            except OSError:add('disconnects')
        def parse_request(self):
            stream=self.rfile;self.rfile=_Headers(stream,16384)
            try:return super().parse_request()
            finally:self.rfile=stream
        def handle_expect_100(self):
            # Do not acknowledge request bodies before authentication/framing checks.
            self.send(417,{'error':'Expect is unsupported'});return False
        def guard(self):
            add('requests')
            hosts=self.headers.get_all('Host',[]);auth=self.headers.get_all('Authorization',[])
            if len(hosts)!=1 or len(auth)>1 or hosts[0].split(':')[0].lower() not in allowed_hosts or self.headers.get_all('Origin'):
                add('invalid_requests');self.send(400,{'error':'Invalid host/origin/header contract'});return False
            if not auth or not secrets.compare_digest(auth[0].encode(),('Bearer '+token).encode()):
                add('auth_rejections');self.send(401,{'error':'Unauthorized'});return False
            if self.headers.get_all('Transfer-Encoding') or len(self.headers.get_all('Content-Length',[]))>1:
                add('invalid_requests');self.send(400,{'error':'Ambiguous request framing'});return False
            return True
        def forward(self,method,body=None):
            self.server.arm(self.connection,upstream_timeout+socket_timeout)
            started=time.monotonic();connection=http.client.HTTPConnection('127.0.0.1',upstream_port,timeout=upstream_timeout)
            timer=None;expired=threading.Event()
            try:
                connection.connect();upstream=connection.sock
                def abort_upstream():expired.set();_abort(upstream)
                timer=threading.Timer(upstream_timeout,abort_upstream);timer.daemon=True;timer.start()
                connection.request(method,self.path,body=body,headers={'Authorization':'Bearer '+backend_token,'Content-Type':'application/json','Connection':'close'})
                response=connection.getresponse();length=response.getheader('Content-Length')
                if response.getheader('Transfer-Encoding') or length is None or not 0<=int(length)<=max_response_bytes:raise ValueError('Upstream framing/size')
                data=response.read(int(length))
                if len(data)!=int(length):raise ValueError('Incomplete upstream response')
                status=response.status
                if status not in (200,400,401,403,404,413,422,429,502,503,504):raise ValueError('Unexpected upstream status')
                if status==200:add('successes')
                else:add('upstream_errors')
                self.send(status,data)
            except (TimeoutError,socket.timeout):add('upstream_timeouts');self.send(504,{'error':'Upstream deadline exceeded'})
            except (OSError,ValueError,http.client.HTTPException):
                if expired.is_set():add('upstream_timeouts');self.send(504,{'error':'Upstream deadline exceeded'})
                else:add('upstream_errors');self.send(502,{'error':'Upstream unavailable'})
            finally:
                if timer is not None:timer.cancel();timer.join()
                connection.close();add('latency_seconds',time.monotonic()-started)
        def do_GET(self):
            if not self.guard():return
            if self.headers.get('Content-Length','0')!='0':self.send(400,{'error':'GET body unsupported'});return
            if self.path=='/metrics':self.send(200,snapshot())
            elif self.path=='/health':self.forward('GET')
            else:self.send(404,{'error':'Not found'})
        def do_POST(self):
            if not self.guard():return
            if self.path!='/predict':self.send(404,{'error':'Not found'});return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=max_body_bytes:self.send(413,{'error':'Body budget exceeded'});return
                if self.headers.get_content_type()!='application/json':self.send(415,{'error':'JSON content type required'});return
                body=self.rfile.read(length)
                if len(body)!=length:raise ValueError('Incomplete body')
            except (OSError,ValueError):add('invalid_requests');self.send(400,{'error':'Incomplete request body'});return
            self.forward('POST',body)
    class Server(ThreadingMixIn,HTTPServer):
        daemon_threads=False;block_on_close=True;request_queue_size=request_slots
        def __init__(self):self.slots=threading.BoundedSemaphore(request_slots);super().__init__((host,port),Handler)
        def get_request(self):
            raw,address=super().get_request();raw.settimeout(socket_timeout);return raw,address
        def process_request(self,request,address):
            add('connections')
            if not self.slots.acquire(blocking=False):add('admission_rejections');self.shutdown_request(request);return
            try:super().process_request(request,address)
            except BaseException:self.slots.release();raise
        def arm(self,sock,seconds):
            _cancel(sock)
            if getattr(sock,'_nil_expired',False):raise TimeoutError('Connection deadline exceeded')
            remaining=min(seconds,sock._nil_deadline-time.monotonic())
            def expire():sock._nil_expired=True;add('connection_timeouts');_abort(sock)
            if remaining<=0:expire();return
            timer=threading.Timer(remaining,expire);timer.daemon=True;sock._nil_timer=timer;timer.start()
        def process_request_thread(self,request,address):
            secure=None
            with lock:
                counts['active_connections']+=1;counts['peak_connections']=max(counts['peak_connections'],counts['active_connections'])
            try:
                secure=context.wrap_socket(request,server_side=True,do_handshake_on_connect=False)
                secure._nil_deadline=time.monotonic()+socket_timeout+upstream_timeout+socket_timeout
                secure.settimeout(socket_timeout);self.arm(secure,socket_timeout);secure.do_handshake()
                self.finish_request(secure,address)
            except (ssl.SSLError,OSError):add('tls_failures')
            except Exception:add('disconnects')
            finally:
                if secure is not None:_cancel(secure)
                self.shutdown_request(secure if secure is not None else request)
                add('active_connections',-1);self.slots.release()
        def handle_error(self,*args):add('disconnects')
        def metrics(self):return snapshot()
    return Server()


class ServingError(RuntimeError):
    def __init__(self,status,attempts,message):self.status,self.attempts=status,attempts;super().__init__(message)


class Client:
    def __init__(self,url,token,*,cafile=None,certfile=None,keyfile=None,max_attempts=3,
                 timeout=15,retry_delay=.05,max_body_bytes=1048576,max_response_bytes=4194304):
        parsed=urlsplit(url)
        if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.path not in ('','/') or parsed.query or parsed.fragment:raise ValueError('Provide an HTTPS origin without credentials/path/query')
        _token(token);_integer(max_attempts,'max_attempts',5);_integer(max_body_bytes,'max_body_bytes',16777216);_integer(max_response_bytes,'max_response_bytes',16777216)
        _seconds(timeout,'timeout');_seconds(retry_delay,'retry_delay')
        port=parsed.port if parsed.port is not None else 443
        _integer(port,'port',65535)
        self.host,self.port,self.token=parsed.hostname,port,token
        self.context=ssl.create_default_context(cafile=cafile)
        if certfile:self.context.load_cert_chain(certfile,keyfile)
        self.attempts,self.timeout,self.delay,self.max_body,self.max_response=max_attempts,timeout,retry_delay,max_body_bytes,max_response_bytes
    def predict(self,inputs):
        data=json.dumps({'inputs':inputs},allow_nan=False).encode()
        if len(data)>self.max_body:raise ValueError('Request exceeds max_body_bytes')
        deadline=time.monotonic()+self.timeout;status=0
        for attempt in range(1,self.attempts+1):
            remaining=deadline-time.monotonic()
            if remaining<=0:raise ServingError(status,attempt-1,'Inference client total deadline exceeded')
            connection=http.client.HTTPSConnection(self.host,self.port,timeout=remaining,context=self.context);timer=None
            try:
                raw=socket.create_connection((self.host,self.port),timeout=remaining)
                remaining=deadline-time.monotonic()
                if remaining<=0:raw.close();raise ServingError(status,attempt,'Inference client total deadline exceeded')
                raw.settimeout(remaining)
                secure=self.context.wrap_socket(raw,server_hostname=self.host,do_handshake_on_connect=False)
                connection.sock=secure
                timer=threading.Timer(remaining,lambda:_abort(secure));timer.daemon=True;timer.start()
                secure.do_handshake()
                connection.request('POST','/predict',body=data,headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json','Connection':'close'})
                response=connection.getresponse();status=response.status;length=response.getheader('Content-Length')
                if response.getheader('Transfer-Encoding') or length is None or not 0<=int(length)<=self.max_response:raise ServingError(status,attempt,'Invalid response framing or size')
                body=response.read(int(length))
                if len(body)!=int(length):raise ServingError(status,attempt,'Incomplete response')
                if status==200:
                    try:result=json.loads(body)
                    except ValueError:raise ServingError(status,attempt,'Invalid JSON response') from None
                    if not isinstance(result,dict) or 'outputs' not in result:raise ServingError(status,attempt,'Invalid prediction schema')
                    return result['outputs']
                if status not in (429,502,503,504):raise ServingError(status,attempt,'Inference request rejected')
            except ssl.SSLCertVerificationError:raise ServingError(0,attempt,'TLS certificate verification failed') from None
            except (OSError,http.client.HTTPException):status=0
            finally:
                if timer is not None:timer.cancel();timer.join()
                connection.close()
            if attempt==self.attempts:raise ServingError(status,attempt,'Inference retry budget exhausted')
            remaining=deadline-time.monotonic()
            if remaining<=0:raise ServingError(status,attempt,'Inference client total deadline exceeded')
            time.sleep(min(self.delay*2**(attempt-1)*secrets.SystemRandom().uniform(.5,1.),remaining))
