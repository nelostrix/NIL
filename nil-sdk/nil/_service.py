"""Explicit reference service lifecycle. No service starts on import."""
import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import signal
import threading


def credential(path):
    path=Path(path)
    if os.name=='posix' and path.stat().st_mode&0o077:raise ValueError('Credential files must not be accessible to group/other users')
    with path.open('rb') as stream:data=stream.read(4097)
    if len(data)>4096:raise ValueError('Credential file exceeds budget')
    return data.decode('ascii').strip()


def run(config,*,ready_file=None):
    from nil.serving import create_isolated_server,create_gateway
    if not isinstance(config,dict):raise ValueError('Service configuration must be an explicit JSON mapping')
    required={'factory','certfile','keyfile','backend_token_file','client_token_file'}
    allowed=required|{'backend_port','gateway_port','host','allowed_hosts','client_ca','request_slots','inference_timeout','startup_timeout','socket_timeout','upstream_timeout'}
    if not required<=set(config) or set(config)-allowed:raise ValueError('Invalid service configuration fields')
    factory=config['factory']
    if not isinstance(factory,str) or ':' not in factory:raise ValueError('Provide an explicit trusted module:factory')
    backend_token=credential(config['backend_token_file']);client_token=credential(config['client_token_file'])
    module,name=factory.split(':',1);model=getattr(import_module(module),name)()
    backend=gateway=None;threads=[];stop=threading.Event();previous={}
    if threading.current_thread() is not threading.main_thread():raise ValueError('Service lifecycle must run on the main thread')
    try:
        for number in (signal.SIGINT,signal.SIGTERM):previous[number]=signal.signal(number,lambda *_:stop.set())
        backend=create_isolated_server(model,token=backend_token,port=config.get('backend_port',0),
            request_slots=config.get('request_slots',8),inference_timeout=config.get('inference_timeout',5),startup_timeout=config.get('startup_timeout',15))
        backend.warmup()
        gateway=create_gateway(upstream_port=backend.server_port,backend_token=backend_token,token=client_token,
            certfile=config['certfile'],keyfile=config['keyfile'],client_ca=config.get('client_ca'),host=config.get('host','127.0.0.1'),
            port=config.get('gateway_port',8443),allowed_hosts=config.get('allowed_hosts',('localhost','127.0.0.1')),
            request_slots=config.get('request_slots',8),socket_timeout=config.get('socket_timeout',2),upstream_timeout=config.get('upstream_timeout',12))
        for server in (backend,gateway):
            thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.05},daemon=True);thread.start();threads.append(thread)
        if ready_file:
            from nil.lifecycle import atomic_json
            atomic_json(ready_file,{'status':'ready','pid':os.getpid(),'backend_port':backend.server_port,'gateway_port':gateway.server_port})
        stop.wait()
    finally:
        # Stop ingress and finish its bounded handlers before shutting down inference.
        for index,server in ((1,gateway),(0,backend)):
            if server is not None:
                if len(threads)>index:server.shutdown()
                server.server_close()
        for thread in threads:thread.join(timeout=2)
        for number,handler in previous.items():signal.signal(number,handler)
        if ready_file:Path(ready_file).unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description='Explicit single-tenant TLS inference service')
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--ready-file',type=Path)
    args=parser.parse_args()
    try:
        if args.config.stat().st_size>65536:raise ValueError('Service configuration exceeds 64 KiB')
        run(json.loads(args.config.read_text()),ready_file=args.ready_file)
    except Exception as error:
        # Do not emit config/credential contents or trusted factory exception details.
        parser.exit(1,'NIL service failed: '+type(error).__name__+'\n')
    return 0
