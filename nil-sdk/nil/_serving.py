"""Bounded, restartable local inference transport; imported only on explicit construction."""
import io
import json
import math
import multiprocessing
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler,HTTPServer
from socketserver import ThreadingMixIn
from nil.runtime import require
from nil._classical_common import positive_int


class WorkerFailure(RuntimeError): pass
class WorkerTimeout(TimeoutError): pass


def _serve_worker(connection,model_bytes,max_inputs,max_outputs):
    import os
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
    try:
        torch=require('torch','torch'); torch.set_num_threads(1)
        # Only the server owner's frozen model archive is deserialized here.
        model=torch.load(io.BytesIO(model_bytes),map_location='cpu',weights_only=False).eval()
        connection.send({'ready':True})
        while True:
            payload=connection.recv()
            try:
                if not isinstance(payload,dict) or set(payload)!={'inputs'}: raise ValueError('schema')
                inputs=torch.tensor(payload['inputs'],dtype=torch.float32)
                if inputs.ndim==0 or not inputs.numel() or inputs.numel()>max_inputs or not torch.isfinite(inputs).all(): raise ValueError('inputs')
                with torch.inference_mode(): output=model(inputs)
                if not isinstance(output,torch.Tensor) or output.numel()>max_outputs or not torch.isfinite(output).all(): raise ValueError('outputs')
                connection.send({'outputs':output.cpu().tolist()})
            except Exception: connection.send({'error':'Input/output or trusted model contract failed'})
    except (EOFError,BrokenPipeError,OSError): pass
    except BaseException:
        try:connection.send({'error':'Trusted model worker initialization/execution failed'})
        except (EOFError,BrokenPipeError,OSError):pass
    finally: connection.close()


class _Worker:
    def __init__(self,archive,startup,deadline,max_inputs,max_outputs):
        self.archive,self.startup,self.deadline,self.max_inputs,self.max_outputs=archive,startup,deadline,max_inputs,max_outputs
        self.process,self.connection=None,None; self.lock=threading.Lock(); self.starts=0
    def _stop(self):
        process=self.process; connection=self.connection; self.process,self.connection=None,None
        if process is not None:
            if process.is_alive(): process.terminate()
            process.join(timeout=2)
            if process.is_alive(): process.kill(); process.join(timeout=2)
            process.close()
        if connection is not None: connection.close()
    def _start(self):
        if self.process is not None and self.process.is_alive(): return
        self._stop(); context=multiprocessing.get_context('spawn'); parent,child=context.Pipe()
        self.process=context.Process(target=_serve_worker,args=(child,self.archive,self.max_inputs,self.max_outputs),daemon=True)
        self.connection=parent
        try:
            self.process.start(); child.close(); self.starts+=1
            if not parent.poll(self.startup) or parent.recv()!={'ready':True}: raise WorkerFailure('Worker startup failed')
        except BaseException: self._stop(); raise
    def predict(self,payload):
        if not self.lock.acquire(blocking=False): return None
        try:
            self._start(); self.connection.send(payload)
            if not self.connection.poll(self.deadline): self._stop(); raise WorkerTimeout('Inference deadline exceeded')
            try: return self.connection.recv()
            except (EOFError,OSError) as error: self._stop(); raise WorkerFailure('Worker exited') from error
        except WorkerTimeout:
            raise
        except (EOFError,BrokenPipeError,OSError) as error:
            self._stop(); raise WorkerFailure('Worker transport failed') from error
        finally: self.lock.release()


def isolated_server(model,*,port,token,request_slots,inference_timeout,startup_timeout,max_body_bytes,max_input_elements,max_output_elements,max_model_bytes):
    import copy
    torch=require('torch','torch')
    for name,value in [('request_slots',request_slots),('max_body_bytes',max_body_bytes),('max_input_elements',max_input_elements),('max_output_elements',max_output_elements),('max_model_bytes',max_model_bytes)]: positive_int(value,name)
    if request_slots>64 or type(port) is not int or not 0<=port<=65535: raise ValueError('Invalid serving port/request slot budget')
    if not isinstance(token,str) or not 16<=len(token)<=4096 or any(not 33<=ord(character)<=126 for character in token): raise ValueError('Provide a printable capability token of at least 16 characters')
    if any(type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=3600 for value in (inference_timeout,startup_timeout)): raise ValueError('Timeouts must be finite, positive and <=3600 seconds')
    if not isinstance(model,torch.nn.Module) or any(p.device.type!='cpu' for p in model.parameters()): raise ValueError('Isolated tensor serving requires an explicit CPU nn.Module')
    archive=io.BytesIO(); torch.save(copy.deepcopy(model).eval(),archive); data=archive.getvalue()
    if len(data)>max_model_bytes: raise ValueError('Frozen model exceeds max_model_bytes')
    worker=_Worker(data,startup_timeout,inference_timeout,max_input_elements,max_output_elements)
    metrics={'requests':0,'rejected':0,'successes':0,'model_errors':0,'timeouts':0,'worker_failures':0,'latency_seconds':0.0}
    metrics_lock=threading.Lock()
    def count(name,value=1):
        with metrics_lock: metrics[name]+=value
    def snapshot():
        with metrics_lock: return {**metrics,'worker_starts':worker.starts,'worker_restarts':max(0,worker.starts-1),'inflight':int(worker.lock.locked())}

    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args): pass
        def send(self,status,body):
            encoded=json.dumps(body,allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(encoded))); self.send_header('Connection','close'); self.end_headers()
            self.close_connection=True
            try: self.wfile.write(encoded)
            except (BrokenPipeError,ConnectionResetError,TimeoutError): pass
        def guard(self):
            count('requests')
            if self.headers.get('Host','').split(':')[0] not in ('127.0.0.1','localhost') or self.headers.get('Origin'):
                count('rejected'); self.send(403,{'error':'Loopback non-browser clients only'}); return False
            if not secrets.compare_digest(self.headers.get('Authorization','').encode(),('Bearer '+token).encode()):
                count('rejected'); self.send(401,{'error':'Unauthorized'}); return False
            return True
        def do_GET(self):
            if not self.guard(): return
            if self.path=='/health': self.send(200,{'status':'ready','inflight':snapshot()['inflight']})
            elif self.path=='/metrics': self.send(200,snapshot())
            else: self.send(404,{'error':'Not found'})
        def do_POST(self):
            if not self.guard(): return
            if self.path!='/predict': self.send(404,{'error':'Not found'}); return
            lengths=self.headers.get_all('Content-Length',[])
            if self.headers.get('Transfer-Encoding') or len(lengths)!=1:
                count('rejected'); self.send(400,{'error':'A single Content-Length is required'}); return
            try:
                length=int(lengths[0])
                if not 0<length<=max_body_bytes: count('rejected'); self.send(413,{'error':'Body budget exceeded'}); return
                body=self.rfile.read(length)
                if len(body)!=length: raise ValueError('Incomplete body')
                payload=json.loads(body)
            except (ValueError,TypeError,TimeoutError,OSError,RecursionError,OverflowError):
                count('rejected'); self.send(422,{'error':'Invalid or incomplete JSON body'}); return
            start=time.monotonic()
            try:
                result=worker.predict(payload)
                if result is None: count('rejected'); self.send(503,{'error':'Inference busy; retry later'}); return
                if 'error' in result: count('model_errors'); self.send(422,{'error':'Input/output or model contract failed'}); return
                count('successes'); self.send(200,result)
            except WorkerTimeout: count('timeouts'); self.send(504,{'error':'Inference deadline exceeded'})
            except Exception: count('worker_failures'); self.send(502,{'error':'Inference worker unavailable'})
            finally: count('latency_seconds',time.monotonic()-start)

    class Server(ThreadingMixIn,HTTPServer):
        daemon_threads=False; block_on_close=True; request_queue_size=request_slots
        def __init__(self): self.slots=threading.BoundedSemaphore(request_slots); super().__init__(('127.0.0.1',port),Handler)
        def get_request(self):
            connection,address=super().get_request(); connection.settimeout(5); return connection,address
        def process_request(self,request,address):
            if not self.slots.acquire(blocking=False):
                count('rejected')
                try: request.sendall(b'HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
                except OSError: pass
                self.shutdown_request(request); return
            try: super().process_request(request,address)
            except BaseException: self.slots.release(); raise
        def process_request_thread(self,request,address):
            try: super().process_request_thread(request,address)
            finally: self.slots.release()
        def server_close(self):
            try: super().server_close()
            finally: worker._stop()
        def warmup(self):
            with worker.lock:worker._start()
        def metrics(self): return snapshot()
    return Server()
