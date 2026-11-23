import concurrent.futures,json,os,subprocess,sys,threading,time
from pathlib import Path
import pytest
torch=pytest.importorskip('torch');torch.set_num_threads(1)
import nil


def test_real_tls_gateway_native_worker_and_warm_load(tls_certificates):
    cert=tls_certificates;model=torch.nn.Linear(1,1)
    with torch.no_grad():model.weight.fill_(2);model.bias.zero_()
    backend=nil.serving.create_isolated_server(model,token='backend-private-capability',inference_timeout=2)
    gateway=nil.serving.create_gateway(upstream_port=backend.server_port,backend_token='backend-private-capability',token='client-private-capability',certfile=cert/'server.crt',keyfile=cert/'server.key',client_ca=cert/'ca.crt')
    threads=[]
    try:
        backend.warmup()
        for server in (backend,gateway):
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();threads.append(thread)
        def predict(value):
            client=nil.serving.client(f'https://localhost:{gateway.server_port}','client-private-capability',cafile=cert/'ca.crt',certfile=cert/'client.crt',keyfile=cert/'client.key',retry_delay=.01,max_attempts=5)
            return client.predict([[value]])
        assert predict(1)==[[2.]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(predict,range(24)))
        assert results==[[[2.*value]] for value in range(24)]
        assert backend.metrics()['successes']==25 and gateway.metrics()['peak_connections']<=16
    finally:
        gateway.shutdown();gateway.server_close();backend.shutdown();backend.server_close()
        for thread in threads:thread.join(timeout=3)


@pytest.mark.skipif(os.name!='posix',reason='POSIX signal lifecycle reference')
def test_service_cli_starts_from_secret_files_and_stops_cleanly(tmp_path,tls_certificates):
    cert=tls_certificates
    factory=tmp_path/'fixture_model.py';factory.write_text('def build():\n import torch\n model=torch.nn.Linear(1,1)\n with torch.no_grad():model.weight.fill_(2);model.bias.zero_()\n return model\n')
    for name in ('backend','client'):
        path=tmp_path/name;path.write_text(name+'-private-capability');path.chmod(0o600)
    config={'factory':'fixture_model:build','certfile':str(cert/'server.crt'),'keyfile':str(cert/'server.key'),'client_ca':str(cert/'ca.crt'),
        'backend_token_file':str(tmp_path/'backend'),'client_token_file':str(tmp_path/'client'),'backend_port':0,'gateway_port':0}
    path=tmp_path/'service.json';path.write_text(json.dumps(config));ready=tmp_path/'ready.json';metrics=tmp_path/'metrics.json'
    source=Path(nil.__file__).resolve().parent.parent
    environment={**os.environ,'PYTHONPATH':os.pathsep.join([str(tmp_path),str(source)])}
    process=subprocess.Popen([sys.executable,'-m','nil.serving','--config',str(path),'--ready-file',str(ready),'--metrics-file',str(metrics),'--metrics-interval','.1'],env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+20
        while not ready.exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.02)
        assert ready.exists(),process.stderr.read().decode() if process.poll() is not None else 'Service startup deadline'
        record=json.loads(ready.read_text())
        client=nil.serving.client(f"https://localhost:{record['gateway_port']}",'client-private-capability',cafile=cert/'ca.crt',certfile=cert/'client.crt',keyfile=cert/'client.key')
        assert client.predict([[3.]])==[[6.]]
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            if metrics.exists() and json.loads(metrics.read_text())['backend']['successes']==1:break
            time.sleep(.02)
        snapshot=json.loads(metrics.read_text())
        assert snapshot['status']=='ready' and snapshot['backend']['successes']==1
        assert snapshot['gateway']['successes']==1 and snapshot['uptime_seconds']>0
        process.terminate();process.wait(timeout=10)
        assert process.returncode==0 and not ready.exists()
        snapshot=json.loads(metrics.read_text())
        assert snapshot['status']=='stopped' and snapshot['gateway']['active_connections']==0
        assert snapshot['backend']['inflight']==0
        assert 'private-capability' not in metrics.read_text()
        stdout,stderr=process.communicate();assert b'private-capability' not in stdout+stderr
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=3)
        process.stdout.close();process.stderr.close()


class StartupFault(torch.nn.Module):
    def __setstate__(self,state):
        super().__setstate__(state)
        if __import__('multiprocessing').current_process().name!='MainProcess':raise RuntimeError('private-startup-secret')
    def forward(self,values):return values


def test_model_startup_errors_are_sanitized_and_worker_is_stopped(capfd):
    server=nil.serving.create_isolated_server(StartupFault(),token='private-startup-capability')
    try:
        with pytest.raises(RuntimeError,match='Worker startup failed'):server.warmup()
        assert 'private-startup-secret' not in ''.join(capfd.readouterr())
        assert server.metrics()['inflight']==0
    finally:server.server_close()


class InferenceFault(torch.nn.Module):
    def forward(self,values):
        if values[0,0].item()==-999:os._exit(17)
        if values[0,0].item()==999:time.sleep(5)
        return values*2


def test_tls_gateway_preserves_worker_crash_timeout_and_restart_contract(tls_certificates):
    cert=tls_certificates
    backend=nil.serving.create_isolated_server(InferenceFault(),token='backend-fault-capability',inference_timeout=.2)
    gateway=nil.serving.create_gateway(upstream_port=backend.server_port,backend_token='backend-fault-capability',token='client-fault-capability',
        certfile=cert/'server.crt',keyfile=cert/'server.key',client_ca=cert/'ca.crt',upstream_timeout=3)
    threads=[]
    try:
        backend.warmup()
        for server in (backend,gateway):
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();threads.append(thread)
        client=nil.serving.client(f'https://localhost:{gateway.server_port}','client-fault-capability',cafile=cert/'ca.crt',certfile=cert/'client.crt',keyfile=cert/'client.key',max_attempts=1,timeout=5)
        with pytest.raises(nil.serving.ServingError) as crashed:client.predict([[-999]])
        assert crashed.value.status==502
        assert client.predict([[3]])==[[6.]]
        with pytest.raises(nil.serving.ServingError) as expired:client.predict([[999]])
        assert expired.value.status==504
        assert client.predict([[4]])==[[8.]]
        assert backend.metrics()['worker_restarts']>=2
    finally:
        gateway.shutdown();gateway.server_close();backend.shutdown();backend.server_close()
        for thread in threads:thread.join(timeout=3)
