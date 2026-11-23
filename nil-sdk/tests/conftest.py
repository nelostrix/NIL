"""Opt-in TLS certificate fixture; no certificate work during unrelated tests."""
import importlib.util
from pathlib import Path
import shutil
import pytest


@pytest.fixture(scope='session')
def tls_certificates(tmp_path_factory):
    if not shutil.which('openssl'):pytest.skip('TLS acceptance requires explicit OpenSSL test tooling')
    helper=Path(__file__).resolve().parents[2]/'scripts/_tls_acceptance.py'
    spec=importlib.util.spec_from_file_location('nil_test_certificates',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.certificates(tmp_path_factory.mktemp('nil-tls'))
