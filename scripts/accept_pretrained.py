"""Real pinned pretrained CPU acceptance; local snapshot only, no downloads.

Stages execute in fresh processes. Optional kernel-policy enforcement requires
an actual <=2GiB/no-swap/one-CPU/128-task Linux unit. This is model-specific
workflow acceptance, not large-model quality or GPU certification.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

TEXTS=[
    'A robot reads its sensors before deciding where to move.',
    'The experiment records measurements and compares the results.',
    'A small computer can process a short sequence of numbers.',
    'The system stores a model and loads the same parameters later.',
    'A telescope collects light from objects in the night sky.',
    'Careful tests check the output before a change is accepted.',
    'The library provides a simple interface for model development.',
    'A program should report failures clearly and release its resources.',
]


def digest_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def prepare(repo,snapshot,revision,max_bytes):
    import re
    if not isinstance(revision,str) or not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Use an immutable40-character model commit')
    manifest=snapshot/'nil-source.json'
    if manifest.exists():
        cached=json.loads(manifest.read_text())
        if cached.get('repo_id')!=repo or cached.get('revision')!=revision:raise FileExistsError('Use a separate snapshot directory for another model/revision')
        if cached['download_bytes']>max_bytes:raise ValueError('Snapshot exceeds explicit download budget')
        if not all(Path(name).name==name and digest_file(snapshot/name)==digest for name,digest in cached['files_sha256'].items()):raise ValueError('Cached snapshot integrity differs')
        return cached
    from huggingface_hub import HfApi,snapshot_download
    info=HfApi().model_info(repo,revision=revision,files_metadata=True,token=False)
    if info.sha!=revision:raise ValueError('Hub resolved a different revision')
    files=[item for item in info.siblings if Path(item.rfilename).name==item.rfilename and (item.rfilename.endswith('.json') or item.rfilename=='model.safetensors')]
    if not any(item.rfilename=='model.safetensors' for item in files) or any(type(item.size) is not int or item.size<0 for item in files):raise ValueError('Require a single-file safetensors model and known file sizes')
    total=sum(item.size for item in files)
    if total>max_bytes:raise ValueError('Snapshot exceeds explicit download budget')
    marker=snapshot/'.nil-preparing.json';expected={'repo_id':repo,'revision':revision}
    if snapshot.exists() and any(snapshot.iterdir()) and (not marker.exists() or json.loads(marker.read_text())!=expected):raise FileExistsError('Snapshot directory is not empty or owned by this preparation')
    snapshot.mkdir(parents=True,exist_ok=True);marker.write_text(json.dumps(expected))
    snapshot_download(repo,revision=revision,local_dir=snapshot,allow_patterns=[item.rfilename for item in files],token=False,max_workers=1)
    hashes={}
    for item in files:
        hashes[item.rfilename]=digest_file(snapshot/item.rfilename)
        if item.lfs and hashes[item.rfilename]!=item.lfs.sha256:raise ValueError('Safetensors digest differs from pinned Hub metadata')
    result={'repo_id':repo,'revision':revision,'download_bytes':total,'files_sha256':hashes,'source_url':'https://huggingface.co/'+repo+'/tree/'+revision}
    manifest.write_text(json.dumps(result,indent=2)+'\n');marker.unlink()
    return result


def frozen_digest(model):
    digest=hashlib.sha256()
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode());digest.update(memoryview(parameter.detach().cpu().contiguous().numpy()).cast('B'))
    return digest.hexdigest()


def worker(stage,snapshot,output):
    import gc
    import math
    import resource
    import torch
    import nil
    torch.set_num_threads(1)
    started=time.monotonic();report={'status':'failed','stage':stage,'versions':{name:version(name) for name in ('nil-sdk','torch','transformers','peft')}}
    try:
        model,tokenizer=nil.generative.load_causal_lm(snapshot,use_safetensors=True,torch_dtype=torch.float32,
            low_cpu_mem_usage=True,attn_implementation='eager',cache_dir=snapshot/'.local-cache')
        model.config.use_cache=False
        count=sum(p.numel() for p in model.parameters())
        if count<100000000:raise ValueError('Real pretrained target must have at least100M parameters')
        data=nil.generative.text_dataset(TEXTS,tokenizer,max_length=32)
        report.update(architecture=type(model).__name__,parameters=count,model_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            nil_path=str(Path(nil.__file__).resolve()),baseline=nil.generative.perplexity(model,data,batch_size=1,device_name='cpu'))
        if stage=='inference':
            generated=nil.generative.generate(model,tokenizer,'The purpose of a test is',max_new_tokens=8,do_sample=False)
            repeated=nil.generative.generate(model,tokenizer,'The purpose of a test is',max_new_tokens=8,do_sample=False)
            if not generated or generated!=repeated:raise ValueError('Deterministic generation did not reproduce')
            report['generation']=generated
        elif stage=='lora':
            adapted=nil.generative.lora(model,rank=4,alpha=8,target_modules=['q_proj','v_proj'])
            report['adapter_parameters']=nil.nn.parameters(adapted)
            frozen=frozen_digest(adapted);initial={name:p.detach().clone() for name,p in adapted.named_parameters() if p.requires_grad}
            with tempfile.TemporaryDirectory(prefix='nil-pretrained-adapter-',dir=output.parent) as directory:
                root=Path(directory)
                trained=nil.train(adapted,data,task='causal_lm',epochs=1,batch_size=1,learning_rate=.001,device='cpu',seed=7,output_dir=root/'training')
                if frozen_digest(trained.model)!=frozen:raise ValueError('LoRA changed frozen base parameters')
                if not any(not torch.equal(initial[name],p) for name,p in trained.model.named_parameters() if p.requires_grad):raise ValueError('LoRA made no parameter update')
                score=nil.generative.perplexity(trained.model,data,batch_size=1,device_name='cpu')
                adapter=root/'adapter';nil.generative.save(trained.model,tokenizer,adapter)
                report['adapter_artifact_bytes']=sum(path.stat().st_size for path in adapter.rglob('*') if path.is_file())
                del trained,adapted,model;gc.collect()
                restored,_=nil.generative.load_causal_lm(adapter,use_safetensors=True,torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='eager')
                restored.config.use_cache=False
                recovered=nil.generative.perplexity(restored,data,batch_size=1,device_name='cpu')
                if not math.isclose(score['nll'],recovered['nll'],rel_tol=1e-6,abs_tol=1e-6):raise ValueError('Adapter-only reload did not reproduce NLL')
                report.update(adapted=score,adapter_reloaded=recovered,frozen_base_unchanged=True,
                    scope='Eight short text windows, eight Adam updates; functional adaptation/reload, not held-out model-quality improvement')
        elif stage=='int8':
            baseline=report['baseline'];quantized=nil.optimization.quantize(model,method='weight_only_int8',exclude=('lm_head',),weight_scheme='per_channel')
            if model.training:raise ValueError('Quantization changed caller mode')
            del model;gc.collect()
            score=nil.generative.perplexity(quantized,data,batch_size=1,device_name='cpu')
            report.update(quantized=score,nll_delta=score['nll']-baseline['nll'],excluded_layers=['lm_head'],linear_weight_scheme='per_channel',method='weight_only_int8')
            if score['nll']>baseline['nll']+.5:raise ValueError('INT8 NLL exceeds declared +0.5 tolerance')
            if not any(getattr(layer,'qweight',None) is not None and layer.qweight.dtype==torch.int8 for layer in quantized.modules()):raise ValueError('No actual INT8 weight storage was constructed')
            with tempfile.TemporaryDirectory(prefix='nil-pretrained-int8-',dir=output.parent) as directory:
                artifact=Path(directory)/'int8.pt';torch.save(quantized.state_dict(),artifact)
                report['serialized_int8_state_bytes']=artifact.stat().st_size
            report.update(quantized=score,nll_delta=score['nll']-baseline['nll'],tolerance_nll_delta=.5,
                scope='CPU per-channel INT8 weight storage, FP32 dequantized Linear computation; embeddings/output head remain floating-point; no INT8 activation/speed claim')
        report['status']='passed'
    except Exception as error:
        report['error']=type(error).__name__+': '+str(error)[-3000:]
    report['seconds']=time.monotonic()-started;report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if report['status']=='passed' else 1


def run(snapshot,output,stages,require_unit_policy):
    output.parent.mkdir(parents=True,exist_ok=True);report={'schema_version':1,'status':'failed','scope':'Pinned >=100M pretrained CPU workflows; no GPU or arbitrary model certification',
        'platform':platform.platform(),'python':platform.python_version(),'stages':{},'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    code=2;group=None
    try:
        source=json.loads((snapshot/'nil-source.json').read_text())
        report['source']=source
        for name,digest in source['files_sha256'].items():
            if Path(name).name!=name or digest_file(snapshot/name)!=digest:raise ValueError('Pinned pretrained snapshot digest differs')
        if require_unit_policy:
            if sys.platform!='linux':raise ValueError('Actual Linux resource policy required')
            relative=Path('/proc/self/cgroup').read_text().split('0::',1)[1].strip().lstrip('/')
            group=Path('/sys/fs/cgroup')/relative
            policy={name:(group/name).read_text().strip() for name in ('memory.max','memory.swap.max','cpu.max','pids.max')}
            quota,period=policy['cpu.max'].split()
            if policy['memory.max']=='max' or not 0<int(policy['memory.max'])<=2147483648 or policy['memory.swap.max']!='0' or quota=='max' or not 0<int(quota)<=int(period) or policy['pids.max']=='max' or not 1<=int(policy['pids.max'])<=128:raise ValueError('Require <=2GiB/no-swap/one-CPU/128-task policy')
            report['kernel_policy']=policy
        code=1
        for stage in stages:
            log=output.with_name(output.stem+'-'+stage+'.log')
            # The parent owns scratch files, so timeout/OOM cannot strand a model
            # checkpoint when the worker's Python finalizers never run.
            with tempfile.TemporaryDirectory(prefix='nil-pretrained-stage-',dir=output.parent) as work:
                child=Path(work)/'report.json'
                command=[sys.executable,str(Path(__file__).resolve()),'--snapshot',str(snapshot),'--worker',stage,'--output',str(child)]
                with log.open('w') as stream:
                    try:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=240);returncode=result.returncode
                    except subprocess.TimeoutExpired:returncode=124
                entry=json.loads(child.read_text()) if child.exists() else {'status':'failed','error':'Worker did not publish a report'}
            entry.update(exit_code=returncode,log=log.name);report['stages'][stage]=entry
            if returncode or entry['status']!='passed':break
        if group:
            report['kernel_usage']={'memory_peak_bytes':int((group/'memory.peak').read_text()),
                'memory_events':{name:int(value) for name,value in (line.split() for line in (group/'memory.events').read_text().splitlines())}}
        passed=len(report['stages'])==len(stages) and all(value['status']=='passed' and not value['exit_code'] for value in report['stages'].values()) and report.get('kernel_usage',{}).get('memory_events',{}).get('oom_kill',0)==0
        report['status']='passed' if passed else 'failed';code=0 if passed else 1
    except (OSError,ValueError,KeyError,IndexError) as error:report['error']=type(error).__name__+': '+str(error)
    report['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report,code


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',type=Path,required=True);parser.add_argument('--output',type=Path)
    parser.add_argument('--prepare-model');parser.add_argument('--revision');parser.add_argument('--max-download-mib',type=int,default=350)
    parser.add_argument('--stages',nargs='+',choices=('inference','lora','int8'),default=['inference','lora','int8'])
    parser.add_argument('--require-unit-policy',action='store_true');parser.add_argument('--worker',choices=('inference','lora','int8'))
    args=parser.parse_args()
    if not 1<=args.max_download_mib<=1024:parser.error('--max-download-mib must be1–1024')
    if args.prepare_model:
        record=prepare(args.prepare_model,args.snapshot.resolve(),args.revision,args.max_download_mib*1048576)
        print(json.dumps({'snapshot':str(args.snapshot.resolve()),'revision':record['revision'],'download_bytes':record['download_bytes']}));return 0
    if args.output is None:parser.error('--output required for acceptance')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.worker:return worker(args.worker,args.snapshot.resolve(),args.output.resolve())
    report,code=run(args.snapshot.resolve(),args.output.resolve(),args.stages,args.require_unit_policy)
    print(json.dumps({'status':report['status'],'stages':{name:value['status'] for name,value in report['stages'].items()},'error':report.get('error')}));return code

if __name__=='__main__':raise SystemExit(main())
