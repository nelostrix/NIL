"""Real filesystem credential refusal and sanitized CLI failure contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from nil._service import credential, run


def test_private_token_file_and_invalid_content(tmp_path):
    path=tmp_path/'credential';path.write_text('distinct-private-capability\n');path.chmod(0o600)
    assert credential(path)=='distinct-private-capability'
    for contents in (b'too-short', b'x'*4097, b'private\nembedded-newline-capability', b'\xff'*20):
        path.write_bytes(contents)
        with pytest.raises((ValueError,UnicodeError)):credential(path)
    with pytest.raises((ValueError,OSError)):credential(tmp_path)


@pytest.mark.skipif(os.name!='posix',reason='POSIX credential permissions and special files')
def test_insecure_permissions_symlinks_and_fifo_are_refused_without_blocking(tmp_path):
    path=tmp_path/'credential';path.write_text('distinct-private-capability');path.chmod(0o644)
    with pytest.raises(ValueError,match='group/other'):credential(path)
    path.chmod(0o600);link=tmp_path/'link';link.symlink_to(path)
    with pytest.raises(OSError):credential(link)
    fifo=tmp_path/'fifo';os.mkfifo(fifo,0o600)
    with pytest.raises(ValueError,match='regular file'):credential(fifo)


def test_service_preflight_refuses_aliased_outputs_invalid_intervals_and_shared_tokens(tmp_path):
    for interval in (True,0,float('nan'),301):
        with pytest.raises(ValueError,match='Metrics interval'):run({},metrics_interval=interval)
    with pytest.raises(ValueError,match='distinct'):run({},ready_file=tmp_path/'out',metrics_file=tmp_path/'out')
    token=tmp_path/'token';token.write_text('shared-private-capability');token.chmod(0o600)
    config=dict(factory='never_import_this_factory:build',certfile='absent',keyfile='absent',backend_token_file=token,client_token_file=token)
    with pytest.raises(ValueError,match='overwrite'):run(config,metrics_file=token)
    assert token.read_text()=='shared-private-capability'
    with pytest.raises(ValueError,match='distinct'):run(config)


def test_cli_failure_does_not_print_config_paths_or_secret_contents(tmp_path):
    path=tmp_path/'private-config-sentinel.json'
    path.write_text(json.dumps({'private-config-sentinel':'private-secret-sentinel'}))
    environment={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
    result=subprocess.run([sys.executable,'-m','nil.serving','--config',str(path)],env=environment,capture_output=True,text=True,timeout=10)
    assert result.returncode==1
    assert result.stdout=='' and result.stderr=='NIL service failed: ValueError\n'
