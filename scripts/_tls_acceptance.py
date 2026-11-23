"""Ephemeral acceptance certificates only; never runtime certificate provisioning."""
from pathlib import Path
import shutil
import subprocess


def certificates(directory):
    openssl=shutil.which('openssl')
    if not openssl:raise RuntimeError('Explicit TLS acceptance tooling requires openssl')
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    def run(*args):subprocess.run([openssl,*map(str,args)],cwd=root,check=True,capture_output=True,timeout=15)
    run('req','-x509','-newkey','ec','-pkeyopt','ec_paramgen_curve:P-256','-nodes','-subj','/CN=NIL-Test-CA','-days','1','-keyout','ca.key','-out','ca.crt','-addext','basicConstraints=critical,CA:TRUE')
    for name,usage in [('server','serverAuth'),('client','clientAuth')]:
        run('req','-new','-newkey','ec','-pkeyopt','ec_paramgen_curve:P-256','-nodes','-subj','/CN='+('localhost' if name=='server' else 'authorized-client'),'-keyout',name+'.key','-out',name+'.csr')
        extension=root/(name+'.ext');extension.write_text('basicConstraints=CA:FALSE\nextendedKeyUsage='+usage+'\n'+('subjectAltName=DNS:localhost,IP:127.0.0.1\n' if name=='server' else ''))
        run('x509','-req','-in',name+'.csr','-CA','ca.crt','-CAkey','ca.key','-CAcreateserial','-days','1','-extfile',extension,'-out',name+'.crt')
    return root
