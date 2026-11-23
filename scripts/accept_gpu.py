"""Measured, real-device GPU acceptance; missing hardware and skips never pass.

python scripts/accept_gpu.py --mode amp --precision fp16 --output gpu.json
python scripts/accept_gpu.py --mode all --output gpu-all.json  # two CUDA/NCCL GPUs
"""
import argparse
from contextlib import redirect_stderr,redirect_stdout
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


class Results:
    def __init__(self): self.tests={}
    def pytest_runtest_logreport(self,report):
        if report.when=='call' or report.failed or report.skipped:
            self.tests[report.nodeid]={'status':report.outcome,'seconds':report.duration,
                'metrics':dict(report.user_properties)}
            if report.failed:self.tests[report.nodeid]['error']=str(report.longrepr)[-8000:]


def run(mode='all',precision='all'):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    report={'schema_version':1,'mode':mode,'precision':precision,'platform':platform.platform(),
        'python':platform.python_version(),'status':'failed','tests':{},'devices':[],
        'scope':'Actual CUDA training acceptance; CPU results and non-CUDA display adapters cannot pass',
        'display_adapters':[]}
    if sys.platform=='linux':
        for card in sorted(Path('/sys/class/drm').glob('card[0-9]*')):
            device=card/'device'
            try:
                report['display_adapters'].append({'vendor_id':(device/'vendor').read_text().strip(),
                    'device_id':(device/'device').read_text().strip(),
                    'driver':(device/'driver').resolve(strict=True).name})
            except OSError:pass
    try:
        import torch
        report.update(torch=str(torch.__version__),cuda_runtime=torch.version.cuda,
                      hip_runtime=getattr(torch.version,'hip',None),cuda_available=torch.cuda.is_available())
        if not torch.cuda.is_available() or torch.version.cuda is None:
            report['error']='Actual CUDA device and explicit CUDA-enabled Torch required'
            report['next_step']='Use a provisioned CUDA machine; this runner does not install drivers/backends or certify AMD/Intel adapters'
            return report,2
        count=torch.cuda.device_count()
        for index in range(count):
            properties=torch.cuda.get_device_properties(index)
            report['devices'].append({'index':index,'name':properties.name,'total_memory_bytes':properties.total_memory,
                'compute_capability':[properties.major,properties.minor]})
        report.update(cuda_device_count=count,nccl_available=torch.distributed.is_nccl_available())
        if mode!='amp' and (count<2 or not report['nccl_available']):report['error']='Two visible CUDA devices and NCCL are required';return report,2
        report['bf16_supported']=torch.cuda.is_bf16_supported()
        if mode in ('amp','all') and precision in ('bf16','all') and not report['bf16_supported']:
            report['error']='Requested BF16 is unsupported on this GPU';return report,2
        try:
            driver=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=3,check=True)
            report['driver_versions']=sorted(set(driver.stdout.strip().splitlines()))
            if not report['driver_versions']:raise ValueError('Empty GPU driver metadata')
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            report['driver_metadata_error']=type(error).__name__;report['error']='Required GPU driver metadata unavailable';return report,2
        import pytest
    except ImportError as error:report['error']=f'Missing explicit acceptance backend: {error}';return report,2
    torch.use_deterministic_algorithms(True)
    root=Path(__file__).resolve().parents[1]; source=root/'nil-sdk'
    if (source/'nil').is_dir():sys.path.insert(0,str(source))
    test=root/'nil-sdk/tests/test_cuda_acceptance.py'; selectors=[]
    if mode in ('amp','all'):
        for name in ('fp16','bf16') if precision=='all' else (precision,):
            selectors.append(str(test)+f'::test_cuda_learning_accumulation_clipping_and_resume[{name}]')
        if precision in ('fp16','all'):selectors.append(str(test)+'::test_cuda_fp16_overflow_skips_update_and_unscales_before_clip')
    for strategy in ('ddp','fsdp'):
        if mode in (strategy,'all'):selectors.append(str(test)+f'::test_cuda_distributed_serial_parity_resume_and_failure[{strategy}]')
    results=Results(); captured=io.StringIO(); started=time.monotonic()
    with redirect_stdout(captured),redirect_stderr(captured): code=pytest.main([*selectors,'-q'],plugins=[results])
    report.update(tests=results.tests,seconds=time.monotonic()-started,log=captured.getvalue()[-32000:])
    passed=code==0 and len(results.tests)==len(selectors) and all(row['status']=='passed' for row in results.tests.values())
    report['status']='passed' if passed else 'failed'
    if not passed:report['error']='One or more required GPU tests failed, skipped or did not execute'
    return report,0 if passed else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['amp','ddp','fsdp','all'],default='all')
    parser.add_argument('--precision',choices=['fp16','bf16','all'],default='all');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report,code=run(args.mode,args.precision)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'error':report.get('error'),'report':str(args.output)}));return code

if __name__=='__main__':raise SystemExit(main())
