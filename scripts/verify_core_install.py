"""Verify a built wheel installs only NIL, offline, with pip's dependency resolver enabled.

Requires Python with venv/ensurepip; no third-party verification packages.
The temporary environment and its files are removed after the check.
"""

import argparse
from email.parser import BytesParser
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import venv
import zipfile


def check_wheel(path):
    """Reject mandatory dependencies, external payloads and bundled native engines."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_paths = [name for name in names if name.endswith('.dist-info/METADATA')]
        if len(metadata_paths) != 1:
            raise ValueError('Expected exactly one wheel METADATA file')
        info = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        if info['Name'] != 'nil-sdk':
            raise ValueError('Expected a nil-sdk wheel')
        metadata_root = metadata_paths[0].split('/')[0]
        for name in names:
            if name.split('/')[0] not in {'nil', metadata_root} or '..' in name.split('/'):
                raise ValueError(f'Unexpected wheel payload: {name}')
            if name.endswith(('.so', '.pyd', '.dll', '.dylib')):
                raise ValueError(f'Bundled native dependency: {name}')
        for requirement in info.get_all('Requires-Dist', []):
            # All dependency declarations must be gated by one explicit extra.
            # Deliberately reject unfamiliar markers instead of guessing their meaning.
            if not re.fullmatch(r"[^;]+;\s*extra\s*==\s*['\"][A-Za-z0-9_-]+['\"]\s*", requirement):
                raise ValueError(f'Base installation could install a dependency: {requirement}')
        return info['Version']


_INVENTORY = """
from importlib.metadata import distributions
import json
print(json.dumps({d.metadata['Name'].lower().replace('_', '-'): d.version
                  for d in distributions()}, sort_keys=True))
"""

_SMOKE = """
import pathlib, sys
def audit(event, args):
    if event.startswith('socket.') or event in {'subprocess.Popen', 'os.system'}:
        raise RuntimeError('Unexpected network or process activity: ' + event)
sys.addaudithook(audit)
import nil
assert nil.training.checkpoint.recover
assert 'site-packages' in str(pathlib.Path(nil.__file__).resolve()), nil.__file__
assert nil.list_models() and nil.help('linear_regression')
for name in ('pipeline', 'fit', 'save', 'load', 'split', 'cross_validate', 'tune',
             'Trainer', 'training', 'preprocessing', 'inspection', 'predict', 'evaluate', 'sklearn',
             'streaming', 'uncertainty', 'imbalance', 'selection', 'execution', 'rl',
             'boosting', 'forecasting', 'ranking', 'recommendation', 'survival', 'datasets', 'native', 'distributed', 'contracts', 'coverage', 'vision'):
    assert getattr(nil, name) is not None, name
assert len(nil.coverage.areas()) == 32
assert nil.coverage.describe('native.tensor')['requires'] == []
assert nil.coverage.availability('neural.supervised')['runtime_verified'] is False
assert nil.vision.ImageTransform.from_dict(nil.vision.ImageTransform().to_dict()).size == (64,64)
model = nil.native.Linear(1,1,seed=0)
result = nil.native.fit(model, [[0.],[1.],[2.]], [[1.],[3.],[5.]], epochs=150, lr=.1)
classifier = nil.native.Linear(2,3,seed=7)
classified = nil.native.fit(classifier, [[1,0],[0,1],[-1,-1]], [0,1,2], epochs=100, lr=.05,
                            loss='cross_entropy', optimizer='adam')
assert classified['history'][-1] < .1
nd = nil.native.Tensor(list(range(24)), shape=(2,3,4), requires_grad=True)
nd.mean(axis=(0,2)).sum().backward()
assert nd.strides == (12,4,1) and nd.grad is not None
sparse = nil.native.SparseCOO([(0,1),(0,1)], [2.,3.], (2,2)).to_csr()
assert sparse.to_dense().tolist() == [[0.,5.],[0.,0.]]
assert '_nil_cpu_kernels' not in sys.modules, 'Core must never compile/load kernels automatically'
assert result['history'][-1] < .001
assert nil.help('native') and nil.help('distributed')
try:
    nil.distributed.train(None,None,None,strategy='fsdp',resume='unused')
except ValueError as error:
    assert 'trusted_resume=True' in str(error)
else:
    raise AssertionError('Sharded trust gate must fail before backend loading')
assert nil.rl.algorithms() == ['a2c', 'ddpg', 'dqn', 'ppo', 'sac', 'td3']
try:
    nil.rl.load('unused')
except ValueError as error:
    assert 'trusted=True' in str(error)
else:
    raise AssertionError('RL trust gate must fail before backend loading')
try:
    nil.model('linear_regression')
except ImportError as error:
    assert 'does not install or download' in str(error), str(error)
else:
    raise AssertionError('Clean core unexpectedly contains sklearn')
for name in ('lag_forecaster', 'query_ranker', 'item_cosine', 'cox_ph', 'kaplan_meier',
             'lightgbm_classifier', 'catboost_classifier'):
    try:
        nil.model(name)
    except ImportError as error:
        assert 'does not install or download' in str(error), (name, str(error))
    else:
        raise AssertionError('Clean core unexpectedly constructed ' + name)
try:
    from nil.sklearn.linear_model import LinearRegression
except ImportError as error:
    assert 'does not install or download' in str(error), str(error)
else:
    raise AssertionError('Mirror unexpectedly supplied an uninstalled backend')
for root in ('numpy', 'torch', 'sklearn', 'xgboost', 'lightgbm', 'catboost', 'lifelines', 'transformers', 'pandas', 'optuna', 'imblearn', 'stable_baselines3', 'gymnasium'):
    assert not any(n == root or n.startswith(root + '.') for n in sys.modules), root
assert not list(pathlib.Path.cwd().iterdir()), 'Import created application state'
print('Installed-wheel imports, help, lazy APIs and missing-backend behavior passed')
"""


def verify(path):
    path = Path(path).resolve(strict=True)
    version = check_wheel(path)
    with tempfile.TemporaryDirectory(prefix='nil-core-install-') as temporary:
        directory = Path(temporary)
        environment = directory / 'env'
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        work = directory / 'work'
        work.mkdir()

        def run(*args):
            try:
                return subprocess.run([str(python), '-I', *args], cwd=work,
                                      text=True, encoding='utf-8', errors='replace', capture_output=True, check=True, timeout=60)
            except subprocess.CalledProcessError as error:
                print(error.stdout or '',end='')
                print(error.stderr or '',end='')
                raise

        before = json.loads(run('-c', _INVENTORY).stdout)
        # Do not use --no-deps: a bad mandatory dependency must fail resolution.
        run('-m', 'pip', '--isolated', '--disable-pip-version-check', '--no-cache-dir',
            'install', '--no-index', '--only-binary=:all:', str(path))
        after = json.loads(run('-c', _INVENTORY).stdout)
        if after != {**before, 'nil-sdk': version} or 'nil-sdk' in before:
            raise ValueError(f'Expected only NIL to be added: before={before}, after={after}')
        print(run('-c', _SMOKE).stdout.strip())
        run('-c', "import nil; e=nil.execution; "
            "assert not e.capabilities()['owned_process_tree'] or "
            "e.run('builtins:sum', [1, 2, 3], job_root='jobs') == 6")
        run('-c', "import nil; e=nil.execution; "
            "assert not e.capabilities()['sampled_group_rss'] or "
            "e.run('time:sleep', .1, limits=e.Limits(group_rss_mb=128), job_root='rss-jobs') is None")
        for command in (['models'], ['help', 'preprocessing'], ['doctor']):
            run('-m', 'nil', *command)
        print(json.dumps({'status': 'passed', 'added_packages': {'nil-sdk': version},
                          'offline': True, 'dependency_resolution': True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    args = parser.parse_args()
    verify(args.wheel)


if __name__ == '__main__':
    main()
