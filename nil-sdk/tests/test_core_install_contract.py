"""Prevent packaging changes from silently adding dependencies to core installs."""

import importlib.util
from pathlib import Path
import zipfile

import pytest


def verifier():
    path = Path(__file__).resolve().parents[2] / 'scripts/verify_core_install.py'
    spec = importlib.util.spec_from_file_location('verify_core_install', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wheel(tmp_path, *, requirements=(), payload='nil/__init__.py'):
    path = tmp_path / 'nil.whl'
    metadata = 'Metadata-Version: 2.1\nName: nil-sdk\nVersion: 0.2.1\n'
    metadata += ''.join(f'Requires-Dist: {item}\n' for item in requirements)
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('nil_sdk-0.2.1.dist-info/METADATA', metadata)
        archive.writestr(payload, '')
    return path


def test_base_wheel_allows_only_explicit_extra_dependencies(tmp_path):
    path = wheel(tmp_path, requirements=['torch>=2.6; extra == "torch"',
                                         'scikit-learn<2,>=1.4; extra == "classical"'])
    assert verifier().check_wheel(path) == '0.2.1'


@pytest.mark.parametrize('requirement', [
    'numpy>=1.26', 'torch; python_version >= "3.10"',
    'numpy; extra == "classical" or python_version >= "3.10"',
    'numpy; extra != "classical"',
])
def test_rejects_dependencies_that_can_apply_to_base_install(tmp_path, requirement):
    with pytest.raises(ValueError, match='Base installation'):
        verifier().check_wheel(wheel(tmp_path, requirements=[requirement]))


@pytest.mark.parametrize('payload', ['numpy/__init__.py', 'nil/engine.so',
                                     'nil/../outside.py'])
def test_rejects_external_or_native_dependency_payload(tmp_path, payload):
    with pytest.raises(ValueError, match='payload|dependency'):
        verifier().check_wheel(wheel(tmp_path, payload=payload))


def test_missing_dependency_does_not_attempt_install(monkeypatch):
    from nil import runtime

    calls = []

    def missing(name):
        calls.append(name)
        raise ModuleNotFoundError(f'No module named {name}', name=name)

    monkeypatch.setattr(runtime, 'import_module', missing)
    with pytest.raises(runtime.MissingDependencyError, match='does not install or download'):
        runtime.require('sklearn', 'classical')
    assert calls == ['sklearn']
