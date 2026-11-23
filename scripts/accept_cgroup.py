"""Run NIL's real kernel gates inside a new, explicitly delegated transient unit.

Use only under the documented systemd-run --user unit named nil-cgroup-accept-*.
This script changes no ancestor/global controller configuration.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report={'schema_version':1,'status':'failed'}
    root=None;anchor=None;enabled=False
    try:
        if sys.platform!='linux':raise RuntimeError('Actual delegated Linux cgroup-v2 required')
        relative=Path('/proc/self/cgroup').read_text().strip().split('0::',1)[1].strip().lstrip('/')
        root=Path('/sys/fs/cgroup')/relative
        if not root.name.startswith('nil-cgroup-accept-') or not root.name.endswith('.service'):raise RuntimeError('Run only in a fresh named nil-cgroup-accept-* transient service')
        if root.stat().st_uid!=os.getuid() or not os.access(root/'cgroup.subtree_control',os.W_OK):raise RuntimeError('Unit has no writable memory delegation')
        if set((root/'cgroup.procs').read_text().split())!={str(os.getpid())}:raise RuntimeError('Delegated root must initially contain only this controller')
        report['unit_policy']={name:(root/name).read_text().strip() for name in ('memory.max','memory.swap.max','cpu.max')}
        if report['unit_policy']['memory.max']!='268435456' or report['unit_policy']['memory.swap.max']!='0':raise RuntimeError('Require outer 256 MiB memory/no-swap policy')
        quota,period=report['unit_policy']['cpu.max'].split()
        if quota=='max' or int(quota)>int(period):raise RuntimeError('Require CPUQuota <=100%')
        anchor=root/'nil-controller';anchor.mkdir()
        (anchor/'cgroup.procs').write_text(str(os.getpid()))
        (root/'cgroup.subtree_control').write_text('+memory');enabled=True
        from nil._execution_resources import validate_cgroup_root
        validate_cgroup_root(root)
        base=Path(__file__).resolve().parents[1]
        execution_report=args.output.with_name(args.output.stem+'-execution.json')
        command=[sys.executable,str(base/'scripts/accept_execution.py'),'--require-rss','--cgroup-root',str(root),'--output',str(execution_report)]
        completed=subprocess.run(command,capture_output=True,text=True,timeout=45)
        report['execution_exit_code']=completed.returncode
        if completed.returncode!=0:raise RuntimeError('Kernel execution gate failed: '+completed.stderr[-2000:])
        report['execution']=json.loads(execution_report.read_text())
        tests=subprocess.run([sys.executable,'-m','pytest',str(base/'nil-sdk/tests/test_execution_resources.py'),'-q'],
            env={**os.environ,'NIL_TEST_CGROUP_ROOT':str(root)},capture_output=True,text=True,timeout=60)
        report['tests_exit_code']=tests.returncode;report['tests_log']=tests.stdout[-12000:]+tests.stderr[-2000:]
        if tests.returncode!=0:raise RuntimeError('Real kernel resource test suite failed')
        report['status']='passed'
        return 0
    except Exception as error:report['error']=f'{type(error).__name__}: {error}';return 2
    finally:
        # Only our new unit subtree is touched. systemd collects the unit on exit.
        if anchor is not None:
            try:
                if enabled:(root/'cgroup.subtree_control').write_text('-memory')
                (root/'cgroup.procs').write_text(str(os.getpid()));anchor.rmdir()
            except OSError as error:report['cleanup_note']=type(error).__name__
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'error':report.get('error')}))

if __name__=='__main__':raise SystemExit(main())
