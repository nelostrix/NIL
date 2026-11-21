"""Fresh-process NIL overhead/workload evidence; never certifies untested hardware."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time


def worker(case):
    import resource
    started = time.perf_counter()
    if case == 'core_import':
        import nil
        result = {'version': nil.__version__, 'optional_backends_loaded': any(name in sys.modules for name in ('torch', 'sklearn', 'numpy'))}
    elif case in {'ridge_native', 'ridge_nil'}:
        import numpy as np
        from sklearn.linear_model import Ridge
        import nil
        rng = np.random.default_rng(7)
        x = rng.normal(size=(100000, 24)); y = x @ rng.normal(size=24)
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
        x = torch.randn(8192, 24); y = x.sum(dim=1, keepdim=True)
        model = torch.nn.Sequential(torch.nn.Linear(24, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        start_train = time.perf_counter()
        with tempfile.TemporaryDirectory() as root:
            trained = nil.train(model, nil.data.tensors(x, y), epochs=2, batch_size=128,
                                device='cpu', num_workers=0, output_dir=root)
            result = {'rows': len(x), 'features': 24, 'epochs': 2, 'batch_size': 128,
                      'training_s': time.perf_counter() - start_train,
                      'losses': [row['loss'] for row in trained.history]}
    else:
        raise ValueError(case)
    result.update(workload_s=time.perf_counter() - started,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform == 'linux' else 1024 ** 2))
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['core_import', 'ridge_native', 'ridge_nil', 'torch_cpu'])
    parser.add_argument('--profile', choices=['core', 'classical', 'neural'], default='core')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker: worker(args.worker); return
    if not 1 <= args.repeats <= 20: parser.error('--repeats must be 1–20')
    if sys.platform != 'linux': parser.error('This evidence harness currently requires Linux ru_maxrss semantics')
    cases = ['core_import'] + (['ridge_native', 'ridge_nil'] if args.profile == 'classical' else ['torch_cpu'] if args.profile == 'neural' else [])
    versions = {}
    for package in ['nil-sdk', 'numpy', 'scikit-learn', 'torch']:
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), platform.processor())
    report = {'schema_version': 1, 'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'platform': platform.platform(), 'cpu': cpu, 'logical_cpus': os.cpu_count(),
              'python': sys.version, 'versions': versions,
              'old_pc_acceptance': False,
              'scope': 'Current host only; one BLAS/Torch thread, fresh child per sample; cold filesystem cache not enforced. RSS is measured, not a memory quota.',
              'cases': {}}
    env = dict(os.environ)
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']: env[key] = '1'
    for case in cases:
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', case],
                                       capture_output=True, text=True, check=True, timeout=120, env=env)
            sample = json.loads(completed.stdout)
            sample['process_wall_s'] = time.perf_counter() - started
            samples.append(sample)
        report['cases'][case] = {'samples': samples, 'median_wall_s': statistics.median(sample['process_wall_s'] for sample in samples),
                                'max_peak_rss_mib': max(sample['peak_rss_mib'] for sample in samples)}
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text)
    print(text)


if __name__ == '__main__': main()
