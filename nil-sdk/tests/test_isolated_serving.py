import concurrent.futures,json,threading,time,urllib.request,urllib.error
import pytest
torch=pytest.importorskip('torch'); torch.set_num_threads(1)
import nil


class FaultModel(torch.nn.Module):
    def forward(self,values):
        if values[0,0].item()==-999: __import__('os')._exit(17)
        if values[0,0].item()==999: time.sleep(5)
        if values[0,0].item()==111: raise RuntimeError('private-internal-detail')
        return values*2


@pytest.fixture
def service():
    server=nil.serving.create_isolated_server(FaultModel(),token='isolated-test-capability',inference_timeout=.3,startup_timeout=30,request_slots=4,max_body_bytes=1024)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    def request(inputs=None,path='/predict',token='isolated-test-capability',headers=None):
        request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}'+path,data=json.dumps({'inputs':inputs}).encode() if inputs is not None else None,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json',**(headers or {})})
        try:
            with urllib.request.urlopen(request,timeout=40) as response: return response.status,json.load(response)
        except urllib.error.HTTPError as error: return error.code,json.load(error) if error.headers.get('Content-Length')!='0' else {}
    try: yield server,request
    finally: server.shutdown(); server.server_close(); thread.join(timeout=3)


def test_isolated_auth_validation_observability_and_sanitized_errors(service):
    server,request=service
    assert request(path='/health',token='wrong')[0]==401
    assert request(path='/health',headers={'Origin':'https://evil.invalid'})[0]==403
    assert request([[1]],headers={'Host':'evil.invalid'})[0]==403
    assert server.metrics()['worker_starts']==0
    assert request([[1]])==(200,{'outputs':[[2.0]]})
    assert request([[float('nan')]])[0]==422
    status,result=request([[111]]); assert status==422 and 'private-internal-detail' not in json.dumps(result)
    assert request(path='/metrics')[1]['successes']==1


def test_real_worker_timeout_and_crash_recover_with_bounded_admission(service):
    server,request=service; assert request([[1]])[0]==200
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        slow=pool.submit(request,[[999]])
        deadline=time.monotonic()+2
        while not server.metrics()['inflight'] and time.monotonic()<deadline: time.sleep(.01)
        assert request([[2]])[0]==503
        assert slow.result()[0]==504
    assert request([[3]])==(200,{'outputs':[[6.0]]})
    assert request([[-999]])[0]==502
    assert request([[4]])==(200,{'outputs':[[8.0]]})
    metrics=server.metrics(); assert metrics['timeouts']==1 and metrics['worker_restarts']>=2 and metrics['inflight']==0
