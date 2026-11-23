"""Require real neural/RL backends before acceptance; never accept skip-only runs."""

import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rl', action='store_true', help='Also require actual Gymnasium/SB3 neural RL tests')
    parser.add_argument('--generative', action='store_true', help='Require local Transformers/PEFT and ONNX runtime acceptance')
    parser.add_argument('--contrib', action='store_true', help='Require actual recurrent/masked SB3 policies (use with --rl)')
    parser.add_argument('--cuda', action='store_true', help='Require an actual CUDA device')
    args = parser.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    modules = ['torch', 'numpy', 'pytest'] + (['gymnasium', 'stable_baselines3'] if args.rl else [])
    if args.generative: modules += ['transformers','peft','onnx','onnxruntime']
    if args.contrib: modules += ['sb3_contrib']
    loaded = {}
    for name in modules:
        try:
            loaded[name] = importlib.import_module(name)
        except ImportError as error:
            print(f'Neural verification requires {name}: {error}', file=sys.stderr)
            return 2
        print(f'{name}: {getattr(loaded[name], "__version__", "available")}')
    if args.cuda and not loaded['torch'].cuda.is_available():
        print('CUDA acceptance requested but no actual CUDA device is available.', file=sys.stderr)
        return 2
    root = Path(__file__).resolve().parents[1]
    if args.cuda:
        return subprocess.call([sys.executable,str(root/'scripts/accept_gpu.py'),'--mode','amp','--output',str(root/'dist/gpu-acceptance.json')],cwd=root,env=os.environ)
    tests = ['test_loader_resume.py', 'test_dl_correctness.py', 'test_development_sdk.py', 'test_quick_development.py', 'test_execution.py','test_native_engine.py','test_distributed_training.py','test_fsdp_training.py','test_isolated_serving.py','test_neural_compatibility.py']
    if args.rl:
        tests += ['test_rl.py', 'test_rl_correctness.py', 'test_rl_control.py','test_offline_neural_rl.py','test_rl_structured.py','test_rl_snapshot.py']
    environment = {**os.environ, 'PYTHONPATH': os.pathsep.join(filter(None, [str(root / 'nil-sdk'), os.environ.get('PYTHONPATH')]))}
    return subprocess.call([sys.executable, '-m', 'pytest', *[str(root / 'nil-sdk/tests' / name) for name in tests], '-q'], cwd=root, env=environment)


if __name__ == '__main__':
    sys.exit(main())
