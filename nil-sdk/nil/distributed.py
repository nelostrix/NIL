"""Opt-in single-host DDP for finite supervised tensors; no workers at import."""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

from nil.runtime import require
from nil._classical_common import positive_int


@dataclass
class DistributedResult:
    model: object
    history: list
    checkpoint: Path
    world_size: int
    export_checkpoint: Path | None = None
    diagnostics: dict | None = None


def train(model,features,targets,*,epochs=1,batch_size=32,world_size=2,lr=.01,seed=0,
          task='regression',output_dir='./nil_distributed',resume=None,callback=None,
          timeout=60,device='cpu',momentum=0,strategy='ddp',trusted_resume=False,
          max_parameters=100_000_000,max_dataset_bytes=256*1024**2):
    """Global batches are partitioned without duplication. Callback/checkpoint run only on rank zero.

    This scoped path uses fp32 supervised SGD, no BatchNorm, dropout or mutable
    forward buffers; checkpoints represent completed epochs. GPU needs real-host
    validation. A failed worker aborts its group; resume the last committed epoch.
    """
    if strategy not in ('ddp','fsdp'): raise ValueError('Choose ddp or fsdp strategy')
    if strategy=='fsdp' and resume and not trusted_resume: raise ValueError('FSDP native partition metadata requires trusted_resume=True')
    import copy,hashlib,math
    torch=require('torch','torch')
    for name,value in [('epochs',epochs),('batch_size',batch_size),('world_size',world_size),('max_parameters',max_parameters),('max_dataset_bytes',max_dataset_bytes)]: positive_int(value,name)
    if world_size>8 or batch_size<world_size: raise ValueError('world_size must be <=8 and <= global batch_size')
    if strategy=='fsdp' and world_size<2: raise ValueError('FSDP requires at least two ranks')
    if type(seed) is not int or not 0<=seed<2**32: raise ValueError('seed must be an integer in [0, 2**32)')
    if device not in ('cpu','cuda') or device=='cuda' and torch.cuda.device_count()<world_size: raise ValueError('Requested distributed devices are unavailable')
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0 or not 0<=momentum<1 or not math.isfinite(lr) or lr<=0: raise ValueError('Invalid timeout/optimizer settings')
    if not isinstance(model,torch.nn.Module): raise TypeError('Pass a torch.nn.Module')
    if sum(p.numel() for p in model.parameters())>max_parameters: raise ValueError('Model exceeds max_parameters')
    if strategy=='fsdp' and (any(not p.requires_grad for p in model.parameters()) or len(list(model.named_parameters(remove_duplicate=False)))!=len(list(model.named_parameters()))): raise ValueError('Scoped FSDP requires trainable, untied parameters')
    if any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.modules.dropout._DropoutNd)) for m in model.modules()) or list(model.buffers()): raise ValueError('Exact reference DDP requires models without batch normalization/dropout/mutable buffers')
    x,y=torch.as_tensor(features).detach().cpu(),torch.as_tensor(targets).detach().cpu()
    if x.is_complex() or y.is_complex() or x.ndim<2 or y.ndim<1 or len(x)!=len(y) or len(x)<world_size or not torch.isfinite(x).all() or not torch.isfinite(y).all(): raise ValueError('Finite matching tensor datasets with enough samples are required')
    if any(p.dtype!=torch.float32 for p in model.parameters()): raise ValueError('Reference DDP requires float32 model parameters')
    if task=='classification' and (y.ndim!=1 or (y<0).any() or not torch.equal(y,y.round())): raise ValueError('Classification needs nonnegative integer class labels')
    normalized_bytes=x.numel()*4+y.numel()*(8 if task=='classification' else 4)
    if max(normalized_bytes,x.numel()*x.element_size()+y.numel()*y.element_size())>max_dataset_bytes: raise ValueError('Data exceeds max_dataset_bytes')
    x=x.float().contiguous().clone(); y=(y.long() if task=='classification' else y.float()).contiguous().clone()
    if task not in ('regression','classification','binary_classification'): raise ValueError('Choose regression, classification or binary_classification')
    # Hash native bytes without introducing NumPy as an additional API dependency.
    digest=hashlib.sha256()
    for values in (x,y): digest.update(memoryview(values.numpy()))
    digest=digest.hexdigest()
    configuration={'batch_size':batch_size,'world_size':world_size,'lr':lr,'momentum':momentum,'seed':seed,'task':task,'device':device,'data_sha256':digest,'x_shape':list(x.shape),'y_shape':list(y.shape)}
    if strategy=='fsdp':
        configuration['strategy']='fsdp'
        configuration['model_schema']=[(name,list(p.shape),str(p.dtype)) for name,p in model.named_parameters()]
        # JSON manifests retain lists, not Python tuples.
        configuration['model_schema']=json.loads(json.dumps(configuration['model_schema']))
        configuration['model_structure']=[[name,type(module).__module__+'.'+type(module).__qualname__,module.extra_repr()] for name,module in model.named_modules()]
        if resume:
            from nil._sharded import inspect
            record,_=inspect(resume,configuration)
            if record['epoch']>epochs: raise ValueError('Checkpoint exceeds requested epochs')
    path=Path(output_dir).resolve(); path.mkdir(parents=True,exist_ok=True)
    # A fresh rendezvous per invocation prevents stale port/store reuse on restart.
    from nil.distributed import _worker
    with tempfile.TemporaryDirectory(prefix='nil-ddp-') as temporary:
        store=Path(temporary)/'rendezvous'
        context=torch.multiprocessing.spawn(_worker,args=(world_size,copy.deepcopy(model).cpu(),x,y,epochs,configuration,str(path),str(resume) if resume else None,callback,str(store),float(timeout)),nprocs=world_size,join=False)
        import time
        deadline=time.monotonic()+timeout
        try:
            while not context.join(timeout=.1):
                if time.monotonic()>deadline: raise TimeoutError('Distributed job exceeded wall deadline')
        except BaseException:
            for process in context.processes:
                if process.is_alive(): process.terminate()
            for process in context.processes:
                process.join(timeout=2)
                if process.is_alive(): process.kill(); process.join()
            if strategy=='fsdp':
                from nil._sharded import cleanup_pending
                cleanup_pending(path)
            raise
    from nil.training import checkpoint
    payload=checkpoint.read(path/'latest.pt')
    checkpoint.restore(payload,model)
    resumable=path/'shards/latest.json' if strategy=='fsdp' else path/'latest.pt'
    if strategy=='fsdp' and resume and record['epoch']==epochs: resumable=Path(resume).resolve()
    return DistributedResult(model,payload['state']['history'],resumable,world_size,path/'latest.pt' if strategy=='fsdp' else None,payload['state'].get('diagnostics'))


def _worker(rank,world,model,x,y,epochs,configuration,path,resume,callback,store,timeout):
    import random
    from datetime import timedelta
    torch=require('torch','torch'); distributed=torch.distributed
    from nil.training import checkpoint
    from nil.training.tasks import resolve
    torch.set_num_threads(1)
    target=torch.device('cpu' if configuration['device']=='cpu' else f'cuda:{rank}')
    if target.type=='cuda':
        torch.cuda.set_device(target); torch.cuda.reset_peak_memory_stats(target)
    distributed.init_process_group('gloo' if target.type=='cpu' else 'nccl',init_method='file://'+store,rank=rank,world_size=world,timeout=timedelta(seconds=timeout))
    try:
        import copy
        model=copy.deepcopy(model).to(target)
        sharded=configuration.get('strategy')=='fsdp'
        if sharded:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            ddp=FSDP(model,device_id=target,use_orig_params=False,limit_all_gathers=True)
        else: ddp=torch.nn.parallel.DistributedDataParallel(model,device_ids=[rank] if target.type=='cuda' else None)
        optimizer=torch.optim.SGD(ddp.parameters(),lr=configuration['lr'],momentum=configuration['momentum'])
        start,history=0,[]
        torch.manual_seed(configuration['seed']+rank); random.seed(configuration['seed']+rank)
        if resume and sharded:
            from nil._sharded import restore
            record,state=restore(resume,ddp,optimizer,configuration,rank)
            torch.set_rng_state(state['torch']); random.setstate(state['python'])
            if target.type=='cuda': torch.cuda.set_rng_state(state['cuda'],target)
            start,history=record['epoch'],record['history']
            diagnostics=record.get('diagnostics',{})
        elif resume:
            payload=checkpoint.read(resume)
            if payload['state'].get('distributed_config')!=configuration: raise ValueError('Distributed resume requires matching data/config/world size')
            if payload['state']['epoch']>=epochs: raise ValueError('Checkpoint already reached requested epochs')
            checkpoint.restore(payload,model,optimizer)
            state=payload['state']['rank_rng'][rank]; torch.set_rng_state(state['torch']); random.setstate(state['python'])
            if target.type=='cuda': torch.cuda.set_rng_state(state['cuda'],target)
            start,history=payload['state']['epoch'],payload['state']['history']
        step=resolve(configuration['task'])
        for epoch in range(start,epochs):
            order=torch.randperm(len(x),generator=torch.Generator().manual_seed(configuration['seed']+epoch))
            total,count=0.,0.
            for offset in range(0,len(x),configuration['batch_size']):
                indices=order[offset:offset+configuration['batch_size']]
                # Merge a tail smaller than world size into the preceding batch.
                if len(indices)<world: break
                if 0<len(order)-offset-len(indices)<world: indices=order[offset:]
                local=indices[rank::world]; batch=(x[local].to(target),y[local].to(target))
                optimizer.zero_grad(set_to_none=True)
                loss=step(ddp,batch); weight=step.normalization_count(batch)
                valid=torch.tensor(float(torch.isfinite(loss).item() and weight>0),device=target)
                distributed.all_reduce(valid,op=distributed.ReduceOp.MIN)
                if not valid.item(): raise ValueError('Invalid distributed loss/normalization')
                denominator=torch.tensor(weight,dtype=torch.float64,device=target); distributed.all_reduce(denominator)
                # DDP averages rank gradients, so undo that average before global normalization.
                (loss*weight*world/denominator.item()).backward()
                finite=torch.tensor(float(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in ddp.parameters())),device=target)
                distributed.all_reduce(finite,op=distributed.ReduceOp.MIN)
                if not finite.item(): raise ValueError('Nonfinite distributed gradients')
                optimizer.step()
                values=torch.tensor([loss.detach().item()*weight,weight],dtype=torch.float64,device=target); distributed.all_reduce(values)
                total+=values[0].item(); count+=values[1].item()
            row={'epoch':epoch+1,'loss':total/count}; history.append(row)
            metric={'rank':rank,'parameter_elements':sum(p.numel() for p in ddp.parameters())}
            if target.type=='cuda':
                torch.cuda.synchronize(target)
                metric.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(target),peak_reserved_bytes=torch.cuda.max_memory_reserved(target))
            metrics=[None]*world; distributed.all_gather_object(metrics,metric)
            diagnostics={'strategy':configuration.get('strategy','ddp'),'device':configuration['device'],'ranks':metrics}
            local_rng={'torch':torch.get_rng_state(),'python':random.getstate()}
            if target.type=='cuda': local_rng['cuda']=torch.cuda.get_rng_state(target)
            rng=[None]*world; distributed.all_gather_object(rng,local_rng)
            from contextlib import nullcontext
            context=FSDP.summon_full_params(ddp,writeback=False,rank0_only=True) if sharded and callback is not None else nullcontext()
            with context:
                if rank==0 and callback is not None:
                    before=[p.detach().clone() for p in model.parameters()]; mode=model.training
                    try: callback(model,dict(row))
                    finally: model.train(mode)
                    if any(not torch.equal(p,old) for p,old in zip(model.parameters(),before)): raise ValueError('Rank-zero callbacks must not change model parameters')
            if sharded:
                from nil._sharded import save
                save(path,ddp,optimizer,configuration,epoch+1,history,local_rng,diagnostics)
            elif rank==0:
                checkpoint.save(Path(path)/'latest.pt',model,optimizer,epoch=epoch+1,history=history,distributed_config=configuration,rank_rng=rng,diagnostics=diagnostics)
            distributed.barrier()
        if sharded:
            with FSDP.summon_full_params(ddp,writeback=False,rank0_only=True):
                if rank==0: checkpoint.save(Path(path)/'latest.pt',model,epoch=epochs,history=history,distributed_config=configuration,diagnostics=diagnostics)
            distributed.barrier()
    finally:
        distributed.destroy_process_group()
