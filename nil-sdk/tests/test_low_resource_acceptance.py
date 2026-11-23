"""Resource-policy refusal and functional benchmark gates; actual units are external acceptance."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


@pytest.mark.parametrize('field,value',[
    ('memory.max','max'),('memory.max','2147483648'),('memory.swap.max','1'),
    ('cpu.max','max 100000'),('cpu.max','200000 100000'),('cpu.max','0 100000'),
    ('pids.max','max'),('pids.max','128'),
])
def test_loose_or_unknown_kernel_policy_is_refused(tmp_path,field,value):
    values={'memory.max':'1073741824','memory.swap.max':'0','cpu.max':'100000 100000','pids.max':'64'}
    values[field]=value
    for name,contents in values.items():(tmp_path/name).write_text(contents)
    with pytest.raises(ValueError):module('accept_low_resource').policy(tmp_path)


def test_explicit_limited_policy_is_recognized(tmp_path):
    values={'memory.max':'1073741824','memory.swap.max':'0','cpu.max':'50000 100000','pids.max':'32'}
    for name,contents in values.items():(tmp_path/name).write_text(contents)
    assert module('accept_low_resource').policy(tmp_path)==values


@pytest.mark.parametrize('case,values',[
    ('native_cpu',{'final_mse':1.}),('native_cpu',{'final_mse':float('nan')}),
    ('ridge_nil',{'training_rmse':1.}),('torch_cpu',{'losses':[1.,1.]}),
    ('torch_cpu',{'losses':[float('nan'),0.]}),('core_import',{'optional_backends_loaded':True}),
])
def test_fast_but_numerically_broken_workload_cannot_pass(case,values):
    sample={'workload_s':.01,'peak_rss_mib':10.,'optional_backends_loaded':False,**values}
    with pytest.raises(ValueError):module('benchmark_workloads').validate_sample(case,sample)


@pytest.mark.skipif(sys.platform!='linux',reason='Linux kernel policy refusal')
def test_unlimited_host_does_not_claim_low_resource_acceptance(tmp_path):
    tool=module('accept_low_resource')
    group=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().split('0::',1)[1].strip().lstrip('/')
    try:tool.policy(group)
    except ValueError:pass
    else:pytest.skip('Already running under the required actual kernel limits')
    output=tmp_path/'report.json'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/accept_low_resource.py'),'--output',str(output)],capture_output=True,text=True,timeout=10)
    assert result.returncode==2
    report=json.loads(output.read_text())
    assert report['status']=='failed' and report['old_pc_acceptance'] is False and 'workloads' not in report
