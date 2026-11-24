"""Fresh-process Python/C matmul+backward parity, latency and RSS evidence.

Explicitly build a trusted kernel first with nil.native.build_cpu_kernel(path).
No numerical package is imported, downloaded or installed by this benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def worker(kernel_path):
    import resource
    from nil import native as n
    kernel=n.load_cpu_kernel(kernel_path) if kernel_path else None
    a=n.Tensor([(i%17-8)/17 for i in range(96*64)],shape=(96,64),requires_grad=True)
    b=n.Tensor([(i%13-6)/13 for i in range(64*48)],shape=(64,48),requires_grad=True)
    start=time.perf_counter();result=a.matmul(b,kernel=kernel);loss=(result**2).mean();loss.backward()
    seconds=time.perf_counter()-start
    print(json.dumps({'seconds':seconds,'loss':loss.item(),'a_gradient_sum':sum(x for row in a.grad for x in row),
        'b_gradient_sum':sum(x for row in b.grad for x in row),'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'optional_backend_loaded':any(name in sys.modules for name in ('torch','numpy','sklearn'))}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.worker:worker(args.kernel);return 0
    if not args.kernel or not args.output:parser.error('--kernel and --output are required')
    if not 1<=args.repeats<=10:parser.error('--repeats must be 1–10')
    if sys.platform!='linux':parser.error('RSS/compiler acceptance currently targets Linux')
    report={'schema_version':1,'platform':platform.platform(),'python':platform.python_version(),
        'scope':'96x64 @64x48 float64 forward and backward on current host; explicit NIL-owned C kernel; no old-PC/GPU certification',
        'kernel_metadata':json.loads(args.kernel.with_suffix(args.kernel.suffix+'.json').read_text()),'cases':{}}
    env={**os.environ}
    from nil import native as n
    env['PYTHONPATH']=str(Path(n.__file__).resolve().parent.parent)
    for name,kernel in (('python',None),('compiled',args.kernel.resolve())):
        samples=[]
        for _ in range(args.repeats):
            command=[sys.executable,'-S',str(Path(__file__).resolve()),'--worker']
            if kernel:command+=['--kernel',str(kernel)]
            result=subprocess.run(command,env=env,capture_output=True,text=True,check=True,timeout=60)
            samples.append(json.loads(result.stdout))
        report['cases'][name]={'samples':samples,'median_seconds':statistics.median(row['seconds'] for row in samples),
            'max_peak_rss_mib':max(row['peak_rss_mib'] for row in samples)}
    reference=report['cases']['python']['samples'][0]
    import math
    passed=all(not row['optional_backend_loaded'] and all(math.isclose(row[key],reference[key],rel_tol=1e-10,abs_tol=1e-12)
        for key in ('loss','a_gradient_sum','b_gradient_sum')) for case in report['cases'].values() for row in case['samples'])
    report['measured_speedup']=report['cases']['python']['median_seconds']/report['cases']['compiled']['median_seconds']
    report['status']='passed' if passed else 'failed'
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('status','measured_speedup')}));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
