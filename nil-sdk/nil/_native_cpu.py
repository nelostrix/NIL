"""Explicit local C compilation/loading. Never called on import or installation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
import tempfile


def build_cpu_kernel(directory):
    if sys.platform!='linux':raise ValueError('Compiled CPU kernel acceptance currently targets Linux only')
    output=Path(directory).resolve();output.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('_kernels')/'cpu.c'
    include=Path(sysconfig.get_paths()['include'])
    if not (include/'Python.h').is_file():raise RuntimeError('Explicit matching Python development headers required')
    target=output/('_nil_cpu_kernels'+sysconfig.get_config_var('EXT_SUFFIX'))
    if target.exists():raise FileExistsError('Use a new build directory; existing kernels are not overwritten')
    compiler=shlex.split(os.environ.get('CC') or sysconfig.get_config_var('CC') or 'cc')
    with tempfile.TemporaryDirectory(prefix='nil-cpu-build-',dir=output) as temporary:
        staged=Path(temporary)/target.name
        command=[*compiler,'-O3','-shared','-fPIC','-ffp-contract=off','-I'+str(include),str(source),'-o',str(staged)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=120)
        if result.returncode:raise RuntimeError('Explicit CPU kernel build failed: '+result.stderr[-4000:])
        # Atomic exclusive publication; simultaneous builders cannot overwrite.
        os.link(staged,target)
    record={'schema_version':1,'platform':sys.platform,'python':sys.version,'abi':sysconfig.get_config_var('SOABI'),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'binary_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'command':command,
            'scope':'NIL-owned Linux float64 matmul kernel; no external numerical runtime; no -march=native'}
    target.with_suffix(target.suffix+'.json').write_text(json.dumps(record,indent=2)+'\n')
    return target


def load_cpu_kernel(path):
    path=Path(path).resolve(strict=True)
    if not path.is_file():raise ValueError('Provide an explicit trusted compiled kernel file')
    if path.name!='_nil_cpu_kernels'+sysconfig.get_config_var('EXT_SUFFIX'):
        raise ValueError('Kernel filename does not match the active Python ABI')
    metadata=json.loads(path.with_suffix(path.suffix+'.json').read_text())
    if metadata.get('abi')!=sysconfig.get_config_var('SOABI') or metadata.get('binary_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Kernel ABI or binary digest differs from build metadata')
    spec=importlib.util.spec_from_file_location('_nil_cpu_kernels',path)
    if spec is None or spec.loader is None:raise ValueError('Invalid compiled kernel path')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if getattr(module,'abi_version',None)!=1 or not callable(getattr(module,'matmul',None)):
        raise ValueError('Unsupported CPU kernel ABI')
    return module
