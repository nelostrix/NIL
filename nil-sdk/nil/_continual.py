"""Continual classification using the shared trainer and checkpoint contracts."""
import hashlib
import json
import math
import re
import uuid
from pathlib import Path
from nil.runtime import require
from nil.lifecycle import atomic_json, _read_json


def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def fit(model,tasks,*,output_dir,classes,epochs=5,batch_size=32,learning_rate=.001,
        replay_capacity=256,seed=0,resume=False,stop_after=None):
    """Reset AdamW each stage; resume only the last committed complete stage.

    tasks is a list of {'id':str,'train':(X,y),'test':(X,y)} with finite CPU
    float32/64 rank2 inputs and int64 labels. Test data never enters replay.
    Caller ensures exclusive output_dir ownership and deterministic model behavior.
    """
    torch=require('torch','torch')
    from nil.learning import ReplayMemory,continual_metrics
    from nil.training import Trainer,TrainingConfig,checkpoint
    if not isinstance(model,torch.nn.Module) or not isinstance(tasks,(tuple,list)) or not 1<=len(tasks)<=32:raise ValueError('Require a Torch model and 1..32 tasks')
    if any(type(v) is not int for v in (classes,epochs,batch_size,replay_capacity,seed)) or not 2<=classes<=256 or not 1<=epochs<=100 or not 1<=batch_size<=1024 or not 0<=replay_capacity<=4096 or not 0<=seed<2**32:raise ValueError('Invalid continual configuration')
    if type(learning_rate) not in (int,float) or not math.isfinite(learning_rate) or not 0<learning_rate<=1 or type(resume) is not bool:raise ValueError('Invalid learning rate/resume')
    if stop_after is not None and (type(stop_after) is not int or not 0<=stop_after<=len(tasks)):raise ValueError('stop_after must be a total stage target')
    parameters=list(model.parameters())
    if not parameters or sum(p.numel() for p in parameters)>1000000 or any(t.device.type!='cpu' for t in [*parameters,*model.buffers()]):raise ValueError('Require a CPU model with at most 1M parameters')
    dtype=parameters[0].dtype
    if dtype not in (torch.float32,torch.float64) or any(p.dtype!=dtype for p in parameters):raise ValueError('Model requires uniform float32/64 parameters')
    validated=[];ids=set();digest=hashlib.sha256();total=0;width=None
    for task in tasks:
        if not isinstance(task,dict) or set(task)!={'id','train','test'} or not isinstance(task['id'],str) or not task['id'] or len(task['id'])>128 or task['id'] in ids:raise ValueError('Require unique task id/train/test')
        ids.add(task['id']);digest.update(_canonical(task['id']).encode());pairs={}
        for split in ('train','test'):
            pair=task[split]
            if not isinstance(pair,(tuple,list)) or len(pair)!=2:raise ValueError('Task split must be (X,y)')
            x,y=pair
            if not isinstance(x,torch.Tensor) or not isinstance(y,torch.Tensor) or x.device.type!='cpu' or y.device.type!='cpu' or x.dtype!=dtype or y.dtype!=torch.int64 or x.ndim!=2 or not 1<=len(x)<=10000 or not 1<=x.shape[1]<=128 or y.shape!=(len(x),) or not torch.isfinite(x).all() or (y<0).any() or (y>=classes).any():raise ValueError('Invalid finite CPU classification split')
            if width is not None and x.shape[1]!=width:raise ValueError('Feature widths must match')
            width=x.shape[1];total+=x.numel()+y.numel()
            if total>1000000:raise ValueError('Task tensor budget exceeds 1M elements')
            # Own input storage so caller mutation cannot alter a validated stage.
            pairs[split]=(x.detach().clone(),y.detach().clone())
            for tensor in pairs[split]:
                digest.update(_canonical([str(tensor.dtype),list(tensor.shape)]).encode());digest.update(tensor.contiguous().numpy().tobytes())
        validated.append(pairs)
    config=dict(schema_version=1,method='cpu_classification_reservoir_v1',task_ids=[t['id'] for t in tasks],data_sha256=digest.hexdigest(),classes=classes,epochs=epochs,batch_size=batch_size,learning_rate=learning_rate,replay_capacity=replay_capacity,seed=seed)
    identity=hashlib.sha256(_canonical(config).encode()).hexdigest();root=Path(output_dir)
    memory=ReplayMemory(max(1,replay_capacity),seed=seed);matrix=[];completed=0
    if resume:
        if (root/'contract.json').stat().st_size>65536 or _read_json(root/'contract.json')!=config:raise ValueError('Continual data/config mismatch')
        if (root/'state.json').stat().st_size>65536:raise ValueError('Stage pointer exceeds 64 KiB')
        pointer=_read_json(root/'state.json')
        if not isinstance(pointer,dict) or set(pointer)!={'completed','checkpoint'} or type(pointer['completed']) is not int or not 0<=pointer['completed']<=len(tasks) or not isinstance(pointer['checkpoint'],str) or not re.fullmatch(rf"stage-{pointer['completed']:03d}-[0-9a-f]{{32}}\.pt",pointer['checkpoint']):raise ValueError('Invalid stage commit')
        last_checkpoint=root/pointer['checkpoint']
        payload=checkpoint.read(last_checkpoint);state=payload['state']
        if state.get('contract_sha256')!=identity or state.get('completed')!=pointer['completed']:raise ValueError('Stage checkpoint contract mismatch')
        completed=state['completed'];matrix=state['accuracy']
        if not isinstance(matrix,list) or len(matrix)!=completed:raise ValueError('Invalid accuracy history')
        for stage,row in enumerate(matrix):
            if not isinstance(row,list) or len(row)!=len(tasks) or any(v is not None for v in row[stage+1:]) or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in row[:stage+1]):raise ValueError('Invalid stage accuracy')
        memory=ReplayMemory.from_state_dict(state['replay'])
        expected_seen=sum(len(validated[i]['train'][0]) for i in range(completed)) if replay_capacity else 0
        if memory.capacity!=max(1,replay_capacity) or memory.seed!=seed or memory.seen!=expected_seen:raise ValueError('Replay contract mismatch')
        for row in memory.records():
            if not isinstance(row,dict) or set(row)!={'x','y'} or not isinstance(row['x'],list) or len(row['x'])!=width or any(type(v) not in (int,float) or not math.isfinite(v) for v in row['x']) or type(row['y']) is not int or not 0<=row['y']<classes:raise ValueError('Invalid replay example')
        if stop_after is not None and stop_after<completed:raise ValueError('Stage target precedes saved progress')
        # All data/config/replay/metric checks precede model or RNG mutation.
        checkpoint.restore(payload,model,restore_rng=True)
    else:
        root.mkdir(parents=True,exist_ok=False);atomic_json(root/'contract.json',config)
        last_checkpoint=root/f'stage-000-{uuid.uuid4().hex}.pt'
        checkpoint.save(last_checkpoint,model,contract_sha256=identity,completed=0,accuracy=[],replay=memory.state_dict())
        atomic_json(root/'state.json',dict(completed=0,checkpoint=last_checkpoint.name))
    target=len(tasks) if stop_after is None else stop_after
    for stage in range(completed,target):
        x,y=validated[stage]['train'];records=memory.records()
        if records:
            x=torch.cat([x,torch.tensor([r['x'] for r in records],dtype=dtype)])
            y=torch.cat([y,torch.tensor([r['y'] for r in records],dtype=torch.int64)])
        config_train=TrainingConfig(epochs=epochs,batch_size=batch_size,learning_rate=learning_rate,device='cpu',seed=(seed+stage)%2**32,output_dir=root/'runs',name=f'continual-{stage:03d}')
        result=Trainer(model,task='classification',config=config_train).fit(torch.utils.data.TensorDataset(x,y))
        if result.status!='completed':raise RuntimeError('Stage did not complete; resume from committed boundary')
        previous=model.training;model.eval();row=[None]*len(tasks)
        try:
            with torch.inference_mode():
                for task_index in range(stage+1):
                    inputs,labels=validated[task_index]['test'];correct=0
                    for first in range(0,len(inputs),batch_size):
                        logits=model(inputs[first:first+batch_size])
                        if logits.shape!=(min(batch_size,len(inputs)-first),classes) or not torch.isfinite(logits).all():raise ValueError('Expected finite [batch,classes] logits')
                        correct+=int((logits.argmax(1)==labels[first:first+batch_size]).sum())
                    row[task_index]=correct/len(inputs)
        finally:model.train(previous)
        if replay_capacity:
            original_x,original_y=validated[stage]['train']
            for features,label in zip(original_x,original_y):memory.add(dict(x=features.tolist(),y=int(label)))
        matrix.append(row);completed=stage+1;path=root/f'stage-{completed:03d}-{uuid.uuid4().hex}.pt'
        # Unique checkpoints leave interrupted publications unreferenced; a
        # retry cannot overwrite the prior committed stage. state.json commits last.
        checkpoint.save(path,model,contract_sha256=identity,completed=completed,accuracy=matrix,replay=memory.state_dict())
        atomic_json(root/'state.json',dict(completed=completed,checkpoint=path.name));last_checkpoint=path
    learned=[row[:completed] for row in matrix]
    return dict(status='completed' if completed==len(tasks) else 'paused',model=model,completed=completed,
                accuracy=matrix,metrics=continual_metrics(learned) if completed else None,
                checkpoint=last_checkpoint,replay=memory.state_dict(),contract=config)
