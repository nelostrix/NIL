"""Actual loopback mTLS + CPU inference load gate, never a public deployment claim.

Explicit test backends: Torch and OpenSSL CLI (ephemeral local certificates only).
"""
import argparse
import concurrent.futures
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import secrets
import shutil
import statistics
import tempfile
import threading
import time


def run(requests=64,concurrency=4,p95_budget_ms=250,require_unit_policy=False,duration=0):
    report={'schema_version':1,'status':'failed','platform':platform.platform(),
        'python':platform.python_version(),'target':'single Linux/CPU operator; loopback mTLS; 1x1 FP32 affine model',
        'scope':'Warm functional/load gate only; no public-ingress, OS-quota, multi-tenant or arbitrary-model SLO certification',
        'requests':requests,'duration_requested_seconds':duration,'concurrency':concurrency,'p95_budget_ms':p95_budget_ms}
    try:
        policy={}
        if sys.platform=='linux':
            relative=Path('/proc/self/cgroup').read_text().strip().split('0::',1)[1].strip().lstrip('/')
            group=Path('/sys/fs/cgroup')/relative
            policy={name:(group/name).read_text().strip() for name in ('memory.max','memory.swap.max','cpu.max','pids.max')}
        report['resource_policy']=policy
        if require_unit_policy:
            quota,period=policy.get('cpu.max','max 1').split()
            if policy.get('memory.max')!='1073741824' or policy.get('memory.swap.max')!='0' or quota=='max' or int(quota)>int(period) or policy.get('pids.max')=='max' or not 1<=int(policy.get('pids.max','0'))<=128:
                report['error']='Required 1 GiB/no-swap/one-CPU/128-task kernel policy unavailable';return report,2
    except (OSError,ValueError,IndexError) as error:
        if require_unit_policy:report['error']='Kernel resource policy cannot be verified';return report,2
        report['resource_policy_error']=type(error).__name__
    try:
        if not shutil.which('openssl'):raise ImportError('Explicit OpenSSL acceptance tooling unavailable')
        import torch
        import nil
    except ImportError as error:report['error']=str(error);return report,2
    torch.set_num_threads(1);report['torch']=str(torch.__version__)
    helper=Path(__file__).with_name('_tls_acceptance.py');spec=importlib.util.spec_from_file_location('nil_acceptance_certs',helper)
    certificates=importlib.util.module_from_spec(spec);spec.loader.exec_module(certificates)
    backend=gateway=None;threads=[]
    with tempfile.TemporaryDirectory(prefix='nil-serving-accept-') as directory:
        cert=certificates.certificates(directory);internal=secrets.token_urlsafe(32);external=secrets.token_urlsafe(32)
        model=torch.nn.Linear(1,1)
        with torch.no_grad():model.weight.fill_(2);model.bias.zero_()
        try:
            backend=nil.serving.create_isolated_server(model,token=internal,inference_timeout=2,startup_timeout=15)
            cold=time.monotonic();backend.warmup();report['worker_startup_seconds']=time.monotonic()-cold
            gateway=nil.serving.create_gateway(upstream_port=backend.server_port,backend_token=internal,token=external,
                certfile=cert/'server.crt',keyfile=cert/'server.key',client_ca=cert/'ca.crt',request_slots=max(8,concurrency))
            for server in (backend,gateway):
                thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.05},daemon=True);thread.start();threads.append(thread)
            def request(index):
                started=time.monotonic()
                try:
                    client=nil.serving.client(f'https://localhost:{gateway.server_port}',external,cafile=cert/'ca.crt',certfile=cert/'client.crt',keyfile=cert/'client.key',max_attempts=5,retry_delay=.01,timeout=5)
                    outputs=client.predict([[float(index)]])
                    if outputs!=[[2.*index]]:raise ValueError('Prediction parity failed')
                    return {'index':index,'seconds':time.monotonic()-started,'status':'passed'}
                except Exception as error:return {'index':index,'seconds':time.monotonic()-started,'status':'failed','error':type(error).__name__}
            # Warm TLS and the model before recording parallel requests.
            warm=request(1)
            if warm['status']!='passed':report['error']='Warmup prediction failed';return report,1
            started=time.monotonic()
            if duration:
                sequence=iter(range(10000));sequence_lock=threading.Lock();until=time.monotonic()+duration
                def sustain():
                    rows=[]
                    while time.monotonic()<until:
                        with sequence_lock:index=next(sequence,None)
                        if index is None:break
                        rows.append(request(index))
                    return rows
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    results=[row for batch in pool.map(lambda _:sustain(),range(concurrency)) for row in batch]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:results=list(pool.map(request,range(requests)))
            report['wall_seconds']=time.monotonic()-started;report['samples']=results;report['completed_requests']=len(results)
            values=sorted(row['seconds'] for row in results);p95=values[max(0,math.ceil(.95*len(values))-1)]
            report.update(median_ms=statistics.median(values)*1000,p95_ms=p95*1000,max_ms=max(values)*1000,
                failures=sum(row['status']!='passed' for row in results),gateway_metrics=gateway.metrics(),backend_metrics=backend.metrics())
            passed=report['failures']==0 and report['p95_ms']<=p95_budget_ms
            if require_unit_policy and report.get('resource_policy'):
                try:
                    report['resource_usage']={'memory_peak_bytes':int((group/'memory.peak').read_text()),
                        'cpu_stat':dict(line.split() for line in (group/'cpu.stat').read_text().splitlines())}
                except OSError:report['resource_usage_note']='Kernel usage counters unavailable'
            report['status']='passed' if passed else 'failed'
            return report,0 if passed else 1
        finally:
            for server in (gateway,backend):
                if server is not None:
                    if threads:server.shutdown()
                    server.server_close()
            for thread in threads:thread.join(timeout=3)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--requests',type=int,default=64);parser.add_argument('--concurrency',type=int,default=4);parser.add_argument('--p95-budget-ms',type=float,default=250)
    parser.add_argument('--duration',type=float,default=0)
    parser.add_argument('--require-unit-policy',action='store_true')
    args=parser.parse_args()
    if not 1<=args.requests<=10000 or not 1<=args.concurrency<=32 or not math.isfinite(args.p95_budget_ms) or not 0<args.p95_budget_ms<=60000:parser.error('Invalid load budgets')
    if not math.isfinite(args.duration) or not 0<=args.duration<=60:parser.error('--duration must be 0-60 seconds')
    report,code=run(args.requests,args.concurrency,args.p95_budget_ms,args.require_unit_policy,args.duration);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({key:report.get(key) for key in ('status','failures','p95_ms','error')}));return code

if __name__=='__main__':raise SystemExit(main())
