"""Test explicitly installed NIL/backends outside the checkout; write bounded evidence.

Does not install anything or add source paths. Profiles are scoped CPU contracts,
not certification of every optional package, model or platform.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


PROFILES={
    'rl': {
        'packages':('torch','numpy','gymnasium','stable-baselines3','sb3-contrib','pytest'),
        'tests':('test_rl_correctness.py','test_rl_control.py','test_rl_snapshot.py','test_offline_neural_rl.py',
                 'test_rl_cql.py','test_rl_centralized.py','test_rl_parallel_masks.py','test_rl_structured.py'),
        'exclude':'',
        'examples':(('rl_workflow.py','--algorithm','ppo'),('rl_workflow.py','--algorithm','sac')),
    },
    'classical': {
        'packages':('scikit-learn','numpy','scipy','pandas','pytest'),
        'tests':('test_classical_workflow.py','test_classical_validation.py','test_classical_inspection.py','test_sklearn_mirror.py'),
        'exclude':'',
        'examples':(('classical_workflow.py',),('sklearn_mirror.py',)),
    },
    'torch': {
        'packages':('torch','numpy','pytest'),
        'tests':('test_dl_correctness.py','test_loader_resume.py','test_isolated_serving.py','test_native_engine.py','test_native_multiclass.py',
                 'test_service_credentials.py','test_tls_gateway.py','test_tls_inference.py',
                 'test_native_nd.py','test_native_sparse.py','test_native_cpu.py'),
        'exclude':'not cuda',
        'examples':(('deep_workflows.py','--architecture','cnn'),('deep_workflows.py','--architecture','lstm'),('deep_workflows.py','--architecture','transformer')),
    },
}


def run(profile,output,*,timeout=300):
    root=Path(__file__).resolve().parents[1];output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    spec=PROFILES[profile];started=time.monotonic()
    report={'schema_version':1,'profile':profile,'status':'failed','python':platform.python_version(),
        'platform':platform.platform(),'versions':{},'steps':[],
        'scope':'Installed-wheel CPU regression and examples; CUDA is explicitly excluded'}
    environment={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'}
    environment.pop('PYTHONPATH',None);environment.pop('PYTEST_ADDOPTS',None)
    code=1
    try:
        report['versions']={name:version(name) for name in ('nil-sdk',)+spec['packages']}
        with tempfile.TemporaryDirectory(prefix='nil-installed-accept-') as temporary:
            work=Path(temporary)
            # Copy fixtures/examples, never SDK implementation. Prepend-mode test
            # modules remain importable by spawned model workers (importlib-only
            # pytest modules cannot be unpickled in a fresh interpreter).
            tests=work/'nil-sdk/tests';tests.mkdir(parents=True)
            for name in (*spec['tests'],'conftest.py'):
                shutil.copy2(root/'nil-sdk/tests'/name,tests/name)
            shutil.copytree(root/'nil-sdk/examples',work/'nil-sdk/examples')
            (work/'scripts').mkdir()
            shutil.copy2(root/'scripts/_tls_acceptance.py',work/'scripts/_tls_acceptance.py')
            def execute(name,arguments):
                log=output.with_name(output.stem+'-'+name+'.log')
                with log.open('wb') as stream:
                    try:
                        result=subprocess.run([sys.executable,'-I',*arguments],cwd=work,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
                        returncode=result.returncode
                    except subprocess.TimeoutExpired:returncode=124
                report['steps'].append({'name':name,'returncode':returncode,'log':log.name})
                if returncode:raise RuntimeError('Acceptance step failed: '+name)
            # A regular wheel must be imported. An editable source install is not acceptance.
            probe=("import json,nil,pathlib; from importlib.metadata import distribution; "
                   "p=pathlib.Path(nil.__file__).resolve(); d=distribution('nil-sdk'); "
                   "editable=json.loads(d.read_text('direct_url.json') or '{}').get('dir_info',{}).get('editable',False); "
                   "print(json.dumps({'nil_path':str(p),'editable':editable})); "
                   "assert not editable and p==d.locate_file('nil/__init__.py').resolve() and "
                   "not p.is_relative_to(pathlib.Path("+repr(str(root))+")), 'Install the regular wheel before verification'")
            execute('installed-import',['-c',probe])
            junit=work/'tests.xml'
            arguments=['-m','pytest','--import-mode=prepend','-o','addopts=','-q','--junitxml='+str(junit)]
            arguments += [str(tests/name) for name in spec['tests']]
            if spec['exclude']:arguments += ['-k',spec['exclude']]
            execute('tests',arguments)
            suites=ET.parse(junit).getroot()
            cases=list(suites.iter('testcase'))
            report['tests']={'collected':len(cases),'skipped':sum(case.find('skipped') is not None for case in cases)}
            if not cases or report['tests']['skipped']:raise RuntimeError('Profile requires positive tests with zero skips')
            for index,(name,*options) in enumerate(spec['examples']):
                execute('example-'+str(index+1),[str(work/'nil-sdk/examples'/name),*options,'--output',str(work/('example-'+str(index+1)))])
            report['status']='passed';code=0
    except Exception as error:
        report['error']=type(error).__name__+': '+str(error)
    report['seconds']=time.monotonic()-started
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report,code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=PROFILES,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout',type=int,default=300)
    args=parser.parse_args()
    if not 1<=args.timeout<=1800:parser.error('--timeout must be 1–1800 seconds per step')
    report,code=run(args.profile,args.output,timeout=args.timeout)
    print(json.dumps(report,indent=2));return code

if __name__=='__main__':raise SystemExit(main())
