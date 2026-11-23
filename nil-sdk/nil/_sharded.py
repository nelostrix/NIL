"""Fixed-world FSDP1 local shards. Only explicit training imports Torch."""
import json
from pathlib import Path
import platform
import re
import uuid
from nil.lifecycle import atomic_json,sha256
from nil.runtime import require


def inspect(path,configuration=None):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>4_000_000: raise ValueError('Invalid sharded manifest')
    record=json.loads(path.read_text())
    if not isinstance(record,dict) or record.get('format')!='nil-fsdp-local-v1': raise ValueError('Unsupported sharded checkpoint')
    directory=record.get('directory','')
    if not isinstance(directory,str) or not re.fullmatch(r'epoch-[0-9]+-[0-9a-f]{32}',directory): raise ValueError('Invalid shard directory')
    root=path.parent/directory
    if root.is_symlink() or root.resolve().parent!=path.parent.resolve(): raise ValueError('Shards must stay in their owned directory')
    config=record.get('configuration')
    if not isinstance(config,dict) or configuration is not None and config!=configuration: raise ValueError('Sharded resume requires identical model/data/config/world size')
    world=config.get('world_size')
    if type(world) is not int or not 2<=world<=8: raise ValueError('Invalid sharded world size')
    if type(record.get('epoch')) is not int or record['epoch']<1 or not isinstance(record.get('history'),list) or len(record['history'])!=record['epoch']: raise ValueError('Invalid sharded epoch history')
    files=record.get('files')
    if not isinstance(files,dict) or set(files)!={f'rank-{rank}.pt' for rank in range(world)}: raise ValueError('Incomplete rank shard set')
    for name,digest in files.items():
        artifact=root/name
        if artifact.is_symlink() or not artifact.is_file() or sha256(artifact)!=digest: raise ValueError('Shard integrity check failed')
    torch=require('torch','torch')
    if record.get('torch')!=str(torch.__version__) or record.get('python')!=platform.python_version(): raise ValueError('Sharded checkpoint requires exact Python/Torch versions')
    return record,root


def save(path,wrapped,optimizer,configuration,epoch,history,rng,diagnostics):
    torch=require('torch','torch'); dist=torch.distributed; rank=dist.get_rank()
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,LocalStateDictConfig
    root=Path(path)/'shards'; root.mkdir(parents=True,exist_ok=True)
    token=[uuid.uuid4().hex if rank==0 else None]; dist.broadcast_object_list(token,src=0)
    temporary=root/('.pending-'+token[0]); committed=root/f'epoch-{epoch}-{token[0]}'
    if rank==0: temporary.mkdir()
    dist.barrier()
    # LOCAL_STATE_DICT retains native partition metadata and actual flat shards.
    with FSDP.state_dict_type(wrapped,StateDictType.LOCAL_STATE_DICT,LocalStateDictConfig(offload_to_cpu=True)):
        payload={'format':'nil-fsdp-rank-v1','rank':rank,'epoch':epoch,'model':wrapped.state_dict(),
                 'optimizer':optimizer.state_dict(),'rng':rng,'configuration':configuration}
        with (temporary/f'rank-{rank}.pt').open('xb') as stream:
            torch.save(payload,stream); stream.flush()
            import os
            os.fsync(stream.fileno())
    dist.barrier()
    if rank==0:
        files={f'rank-{part}.pt':sha256(temporary/f'rank-{part}.pt') for part in range(dist.get_world_size())}
        record={'format':'nil-fsdp-local-v1','directory':committed.name,'configuration':configuration,
                'epoch':epoch,'history':history,'diagnostics':diagnostics,'files':files,'torch':str(torch.__version__),'python':platform.python_version()}
        temporary.rename(committed)
        from nil.training.checkpoint import _sync_directory
        _sync_directory(root)
        # Manifest last: an interrupted publication leaves the preceding epoch authoritative.
        previous=root/'latest.json'
        if previous.exists(): record['previous']=json.loads(previous.read_text()).get('directory')
        atomic_json(previous,record); _sync_directory(root)
        keep={committed.name,record.get('previous')}
        import shutil
        for candidate in root.iterdir():
            if re.fullmatch(r'epoch-[0-9]+-[0-9a-f]{32}',candidate.name) and candidate.name not in keep and candidate.is_dir() and not candidate.is_symlink(): shutil.rmtree(candidate)
    dist.barrier()
    return root/'latest.json'


def restore(path,wrapped,optimizer,configuration,rank):
    torch=require('torch','torch')
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,LocalStateDictConfig
    record,root=inspect(path,configuration)
    # Native ShardedTensor metadata is executable pickle: caller must opt into trusted_resume.
    payload=torch.load(root/f'rank-{rank}.pt',map_location=next(wrapped.parameters()).device,weights_only=False)
    if payload.get('format')!='nil-fsdp-rank-v1' or payload.get('rank')!=rank or payload.get('epoch')!=record['epoch'] or payload.get('configuration')!=configuration: raise ValueError('Rank shard metadata mismatch')
    with FSDP.state_dict_type(wrapped,StateDictType.LOCAL_STATE_DICT,LocalStateDictConfig(offload_to_cpu=True)):
        wrapped.load_state_dict(payload['model'],strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    return record,payload['rng']


def cleanup_pending(path):
    import shutil
    root=Path(path)/'shards'
    if not root.is_dir(): return
    for candidate in root.iterdir():
        if re.fullmatch(r'\.pending-[0-9a-f]{32}',candidate.name) and candidate.is_dir() and not candidate.is_symlink(): shutil.rmtree(candidate)
