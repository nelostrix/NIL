"""Fresh-process NIL overhead/workload evidence; never certifies untested hardware."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import math
import subprocess
import sys
import tempfile
import time


def worker(case,size="medium"):
    import resource
    started = time.perf_counter()
    if case == 'core_import':
        import nil
        result = {'version': nil.__version__, 'optional_backends_loaded': any(name in sys.modules for name in ('torch', 'sklearn', 'numpy'))}
    elif case=='native_cpu':
        import nil
        count,epochs=(65,150) if size=='small' else (257,100)
        x=[[2*i/(count-1)-1] for i in range(count)]; y=[[2*row[0]+1] for row in x]
        model=nil.native.Linear(1,1,seed=7); begin=time.perf_counter()
        fitted=nil.native.fit(model,x,y,epochs=epochs,lr=.15)
        result={'rows':count,'features':1,'epochs':epochs,'fit_s':time.perf_counter()-begin,
                'final_mse':fitted['history'][-1],'optional_backends_loaded':any(name in sys.modules for name in ('torch','sklearn','numpy'))}
    elif case in {'ridge_native', 'ridge_nil'}:
        import numpy as np
        from sklearn.linear_model import Ridge
        import nil
        rng = np.random.default_rng(7)
        rows,features=(2048,8) if size=="small" else (100000,24)
        x = rng.normal(size=(rows,features)); y = x @ rng.normal(size=features)
        model = Ridge()
        start_fit = time.perf_counter()
        with tempfile.TemporaryDirectory() as root:
            if case == 'ridge_native': model.fit(x, y)
            else: nil.fit(model, x, y, output_dir=root)
            fit = time.perf_counter() - start_fit
            start_predict = time.perf_counter(); predicted = model.predict(x)
            predict = time.perf_counter() - start_predict
            result = {'rows': len(x), 'features': x.shape[1], 'fit_s': fit, 'predict_s': predict,
                      'training_rmse': float(np.sqrt(np.mean((predicted - y) ** 2)))}
    elif case == 'torch_cpu':
        import torch
        import nil
        torch.set_num_threads(1); torch.manual_seed(7)
        rows,features,width,batch=(512,8,16,64) if size=="small" else (8192,24,64,128)
        x = torch.randn(rows, features); y = x.sum(dim=1, keepdim=True)
        model = torch.nn.Sequential(torch.nn.Linear(features, width), torch.nn.ReLU(), torch.nn.Linear(width, 1))
        start_train = time.perf_counter()
        with tempfile.TemporaryDirectory() as root:
            trained = nil.train(model, nil.data.tensors(x, y), epochs=2, batch_size=batch,
                                device='cpu', num_workers=0, output_dir=root)
            result = {'rows': len(x), 'features': features, 'epochs': 2, 'batch_size': batch,
                      'training_s': time.perf_counter() - start_train,
                      'losses': [row['loss'] for row in trained.history]}
    else:
        raise ValueError(case)
    result.update(workload_s=time.perf_counter() - started,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform == 'linux' else 1024 ** 2))
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['core_import','native_cpu', 'ridge_native', 'ridge_nil', 'torch_cpu'])
    parser.add_argument('--profile', choices=['core', 'classical', 'neural'], default='core')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--size',choices=['small','medium'],default='medium')
    parser.add_argument('--timeout',type=float,default=120)
    parser.add_argument('--budgets',type=Path,help='Explicit JSON case budgets (max_wall_s, max_peak_rss_mib)')
    parser.add_argument('--host-label',default='current-host')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0<args.timeout<=3600: parser.error('--timeout must be finite, positive and <=3600')
    if args.worker: worker(args.worker,args.size); return 0
    if not 1 <= args.repeats <= 20: parser.error('--repeats must be 1–20')
    if sys.platform != 'linux': parser.error('This evidence harness currently requires Linux ru_maxrss semantics')
    cases = ['core_import','native_cpu'] + (['ridge_native', 'ridge_nil'] if args.profile == 'classical' else ['torch_cpu'] if args.profile == 'neural' else [])
    budgets={}
    if args.budgets:
        try:
            budgets=json.loads(args.budgets.read_text())
            if not isinstance(budgets,dict) or set(budgets)-set(cases): raise ValueError('Budget cases must belong to the selected profile')
            for case,limits in budgets.items():
                if not isinstance(limits,dict) or not limits or set(limits)-{'max_wall_s','max_peak_rss_mib'}: raise ValueError('Unknown budget metric')
                if any(type(value) not in (int,float) or not math.isfinite(value) or value<=0 for value in limits.values()): raise ValueError('Budgets must be finite positive numbers')
        except (OSError,ValueError) as error: parser.error(str(error))
    versions = {}
    for package in ['nil-sdk', 'numpy', 'scikit-learn', 'torch']:
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), platform.processor())
    report = {'schema_version': 2, 'profile':args.profile,'size':args.size,'host_label':args.host_label,'budgets':budgets, 'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'platform': platform.platform(), 'cpu': cpu, 'logical_cpus': os.cpu_count(),
              'python': sys.version, 'versions': versions,
              'old_pc_acceptance': False,
              'scope': 'Current host only; one BLAS/Torch thread, fresh child per sample; cold filesystem cache not enforced. RSS is measured, not a memory quota.',
              'cases': {}}
    env = dict(os.environ)
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']: env[key] = '1'
    for case in cases:
        samples=[]; failures=[]
        for index in range(args.repeats):
            started=time.perf_counter()
            try:
                completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',case,'--size',args.size],
                    capture_output=True,text=True,check=True,timeout=args.timeout,env=env)
                sample=json.loads(completed.stdout)
                sample['process_wall_s']=time.perf_counter()-started
                samples.append(sample)
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,ValueError) as error:
                failures.append({'sample':index,'error':type(error).__name__,'detail':str(error)[-2000:]})
        summary={'samples':samples,'failures':failures,'status':'failed' if failures else 'passed'}
        if samples:
            summary.update(median_wall_s=statistics.median(row['process_wall_s'] for row in samples),
                max_wall_s=max(row['process_wall_s'] for row in samples),
                max_peak_rss_mib=max(row['peak_rss_mib'] for row in samples))
            violations=[]
            for name,threshold in budgets.get(case,{}).items():
                if summary[name]>threshold: violations.append({'metric':name,'actual':summary[name],'budget':threshold})
            summary['budget_violations']=violations
            if violations: summary['status']='failed'
        report['cases'][case]=summary
    if all(report['cases'].get(case,{}).get('status')=='passed' for case in ('ridge_native','ridge_nil')):
        native=statistics.median(row['fit_s'] for row in report['cases']['ridge_native']['samples'])
        wrapped=statistics.median(row['fit_s'] for row in report['cases']['ridge_nil']['samples'])
        report['ridge_wrapper_overhead']={'native_median_fit_s':native,'nil_median_fit_s':wrapped,
            'fit_ratio':wrapped/native,'added_fit_seconds':wrapped-native,
            'scope':'Independent fresh processes; small timings include noise and explicit NIL artifact/run recording'}
    report['status']='passed' if all(case['status']=='passed' for case in report['cases'].values()) else 'failed'
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text)
    print(text)
    return 0 if report["status"]=="passed" else 1


if __name__ == '__main__': raise SystemExit(main())
