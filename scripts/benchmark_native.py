"""Dependency-free reference subset: cold child import/training time and RSS."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def worker():
    import resource
    start=time.perf_counter()
    import nil
    import_time=time.perf_counter()-start
    x=[[i/32] for i in range(-32,33)]; y=[[2*row[0]+1] for row in x]
    model=nil.native.Linear(1,1,seed=7); start=time.perf_counter()
    result=nil.native.fit(model,x,y,epochs=150,lr=.15)
    print(json.dumps({'import_seconds':import_time,'fit_seconds':time.perf_counter()-start,
        'final_mse':result['history'][-1],'rows':65,'epochs':150,'parameters':2,
        'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'backend_loaded':any(name in sys.modules for name in ('numpy','torch','sklearn'))}))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--worker',action='store_true'); parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.worker: worker(); return
    if not args.output: parser.error('--output is required')
    samples=[]
    for _ in range(5):
        started=time.perf_counter(); result=subprocess.run([sys.executable,'-S',str(Path(__file__).resolve()),'--worker'],check=True,capture_output=True,text=True)
        row=json.loads(result.stdout); row['wall_seconds']=time.perf_counter()-started; samples.append(row)
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor())
    report={'scope':'small dense float64 CPU reference engine; current Linux host only, warm filesystem cache possible; no old-PC acceptance or speedup claim',
        'platform':platform.platform(),'cpu':cpu,'python':platform.python_version(),'samples':samples,
        'target':'65x1 linear least squares, 150 full-batch SGD steps; no numerical backend import; final MSE <1e-6',
        'median_fit_seconds':statistics.median(row['fit_seconds'] for row in samples),
        'median_peak_rss_mib':statistics.median(row['peak_rss_mib'] for row in samples)}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+'\n')
    assert all(not row['backend_loaded'] and row['final_mse']<1e-6 for row in samples)
    print(json.dumps({key:report[key] for key in ('median_fit_seconds','median_peak_rss_mib')}))

if __name__=='__main__': main()
