"""Actual Linux low-resource gate; never represents modern hardware as an old PC.

Run inside an explicitly limited transient user unit (see docs/HARDWARE_ACCEPTANCE.md).
The script reads the kernel policy; it does not change system settings or install
packages. Only small installed-NIL workloads and explicit backends are exercised.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def policy(root):
    values={name:(root/name).read_text().strip() for name in ('memory.max','memory.swap.max','cpu.max','pids.max')}
    if values['memory.max']=='max' or not 0<int(values['memory.max'])<=1073741824:
        raise ValueError('Require a hard memory limit <=1 GiB')
    if values['memory.swap.max']!='0':raise ValueError('Require swap disabled for this unit')
    quota,period=values['cpu.max'].split()
    if quota=='max' or not 0<int(quota)<=int(period):raise ValueError('Require positive CPUQuota <=100%')
    if values['pids.max']=='max' or not 1<=int(values['pids.max'])<=64:raise ValueError('Require TasksMax <=64')
    return values


def counters(root):
    return {'memory_peak_bytes':int((root/'memory.peak').read_text()),
            'memory_events':{key:int(value) for key,value in (row.split() for row in (root/'memory.events').read_text().splitlines())},
            'cpu_stat':{key:int(value) for key,value in (row.split() for row in (root/'cpu.stat').read_text().splitlines())}}


def run(profile,output,repeats=3):
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    report={'schema_version':1,'status':'failed','profile':profile,'old_pc_acceptance':False,
        'scope':'Current host under actual kernel limits: <=1 CPU, <=1 GiB RAM, zero swap, <=64 tasks. Not old CPU/ISA/storage emulation.',
        'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    code=2;group=None
    try:
        if sys.platform!='linux':raise ValueError('Actual Linux cgroup-v2 required')
        relative=Path('/proc/self/cgroup').read_text().split('0::',1)[1].strip().lstrip('/')
        group=Path('/sys/fs/cgroup')/relative
        report['kernel_policy']=policy(group);before=counters(group)
        base=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='nil-low-resource-') as temporary:
            benchmark=Path(temporary)/'workloads.json'
            command=[sys.executable,str(base/'scripts/benchmark_workloads.py'),'--profile',profile,'--size','small',
                     '--repeats',str(repeats),'--timeout','60','--installed',
                     '--budgets',str(base/'benchmarks/budgets'/f'{profile}-small.json'),
                     '--host-label','current-host-1cpu-1GiB-no-swap','--output',str(benchmark)]
            report['benchmark_started']=True
            completed=subprocess.run(command,capture_output=True,text=True,timeout=300)
            report['benchmark_exit_code']=completed.returncode
            if completed.returncode:report['benchmark_diagnostics']=(completed.stdout+completed.stderr)[-3000:]
            if not benchmark.exists():raise RuntimeError('Benchmark did not publish evidence')
            report['workloads']=json.loads(benchmark.read_text())
        after=counters(group)
        report['kernel_usage']=after
        report['oom_kill_delta']=after['memory_events'].get('oom_kill',0)-before['memory_events'].get('oom_kill',0)
        if completed.returncode!=0 or report['workloads']['status']!='passed' or report['oom_kill_delta']:
            raise RuntimeError('Workload, latency/RSS budget or kernel OOM acceptance failed')
        report['status']='passed';code=0
    except (OSError,ValueError,IndexError,KeyError,RuntimeError,subprocess.SubprocessError) as error:
        report['error']=type(error).__name__+': '+str(error)
        if report.get('benchmark_started'):code=1
    report['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report,code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('core','classical','neural'),default='core')
    parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.repeats<=10:parser.error('--repeats must be 1–10')
    report,code=run(args.profile,args.output,args.repeats)
    print(json.dumps({key:report.get(key) for key in ('status','profile','error','old_pc_acceptance')}))
    return code

if __name__=='__main__':raise SystemExit(main())
