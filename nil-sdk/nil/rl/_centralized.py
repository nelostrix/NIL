"""Offline centralized-value AWR; fixed aligned agents, decentralized discrete actors.

This is an explicit supervised-return/offline algorithm, not MAPPO or an online
centralized controller. Requires optional CPU Torch, never imports on core NIL load.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from nil.runtime import require
from nil.training import checkpoint
from nil._classical_common import positive_int
from ._offline import _dataset

torch=require('torch','torch')


def _network(inputs,outputs,hidden):
    dimensions=[inputs,*hidden,outputs];layers=[]
    for index,(left,right) in enumerate(zip(dimensions,dimensions[1:])):
        layers.append(torch.nn.Linear(left,right))
        if index<len(dimensions)-2:layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)


class Actor:
    def __init__(self,network,seed):
        self.policy=network;self.device=torch.device('cpu');self.action_generator=torch.Generator().manual_seed(seed)
    def predict(self,observations,*,deterministic=True,action_masks=None):
        values=torch.as_tensor(observations,dtype=torch.float32);single=values.ndim==1
        if single:values=values[None,:]
        if values.ndim!=2 or values.shape[1]!=self.policy[0].in_features or not torch.isfinite(values).all():raise ValueError('Expected finite local observation rows')
        with torch.inference_mode():
            logits=self.policy(values)
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite decentralized actor output')
            if action_masks is not None:
                mask=torch.as_tensor(action_masks)
                if single and mask.ndim==1:mask=mask[None,:]
                if mask.shape!=logits.shape or not ((mask==0)|(mask==1)).all() or not mask.bool().any(dim=1).all():raise ValueError('Invalid action masks')
                logits=logits.masked_fill(~mask.bool(),-torch.inf)
            if deterministic:actions=logits.argmax(dim=1)
            else:actions=torch.multinomial(logits.softmax(dim=1),1,generator=self.action_generator).squeeze(1)
        values=actions.numpy();return (values[0] if single else values),None


class Model(torch.nn.Module):
    def __init__(self,config):
        super().__init__();specs=config['agents'];hidden=config['hidden']
        if not 1<=len(specs)<=16 or not isinstance(hidden,list) or len(hidden)>3 or any(type(n) is not int or not 1<=n<=256 for n in hidden):raise ValueError('Centralized model exceeds agent/hidden budget')
        if any(not isinstance(name,str) or not name or len(name)>128 or type(inputs) is not int or not 1<=inputs<=4096 or type(actions) is not int or not 2<=actions<=4096 for name,inputs,actions in specs):raise ValueError('Invalid centralized agent shape')
        total=sum(spec[1] for spec in specs)
        if total>4096 or len({spec[0] for spec in specs})!=len(specs):raise ValueError('Invalid joint feature/agent budget')
        count=sum(sum((a+1)*b for a,b in zip([inputs,*hidden,outputs],[*hidden,outputs])) for _,inputs,outputs in specs)
        count+=sum((a+1)*b for a,b in zip([total,*hidden,1],[*hidden,1]))
        if count>1000000:raise ValueError('Centralized model exceeds parameter budget')
        with torch.random.fork_rng():
            torch.manual_seed(config['seed'])
            self.actors=torch.nn.ModuleList([_network(inputs,outputs,hidden) for _,inputs,outputs in specs])
            self.critic=_network(total,1,hidden)


@dataclass
class CentralizedResult:
    policies: dict
    critic: object
    history: list
    checkpoint: Path
    status: str


def _result(model,config,history,path,status,rng_states=None):
    policies={spec[0]:Actor(actor,config['seed']+index+1) for index,(spec,actor) in enumerate(zip(config['agents'],model.actors))}
    if rng_states:
        for name,policy in policies.items():policy.action_generator.set_state(rng_states[name])
    model.eval();return CentralizedResult(policies,model.critic,history,path,status)


def fit(datasets,*,action_counts,returns,steps=500,batch_size=64,hidden=(32,32),lr=.001,
        temperature=1.,max_weight=20.,seed=0,max_dataset_bytes=64000000,
        output_dir='./nil_centralized',resume=None,cancel_event=None):
    if not isinstance(datasets,dict) or not isinstance(action_counts,dict) or not 1<=len(datasets)<=16 or set(datasets)!=set(action_counts):raise ValueError('Provide 1–16 fixed agent-keyed datasets and action counts')
    for name,value in (('steps',steps),('batch_size',batch_size),('max_dataset_bytes',max_dataset_bytes)):positive_int(value,name)
    if steps>100000 or batch_size>100000 or type(seed) is not int or not 0<=seed<2**32:raise ValueError('Invalid centralized steps/batch/seed')
    import math
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (lr,temperature,max_weight)) or lr<=0 or temperature<=0 or not 1<=max_weight<=100:raise ValueError('Invalid AWR optimizer/weight configuration')
    records={};specs=[];count=None;digests=[];used=0
    for name,data in datasets.items():
        normalized,continuous,actions,digest=_dataset(data,'behavior_cloning',action_counts[name],max_dataset_bytes)
        used+=sum(value.numel()*value.element_size() for value in normalized.values())
        if used>max_dataset_bytes:raise ValueError('Joint dataset exceeds max_dataset_bytes')
        if continuous:raise ValueError('Centralized AWR currently supports discrete actors only')
        if count is None:count=len(normalized['observations'])
        if len(normalized['observations'])!=count:raise ValueError('Centralized datasets must have aligned row counts')
        records[name]=normalized;specs.append([name,normalized['observations'].shape[1],actions]);digests.append(digest)
    target=torch.as_tensor(returns,dtype=torch.float32).detach().cpu()
    if target.shape!=(count,) or not torch.isfinite(target).all():raise ValueError('Provide one finite explicit team return per aligned row')
    total=sum(value.numel()*value.element_size() for record in records.values() for value in record.values())+target.numel()*4
    if total>max_dataset_bytes:raise ValueError('Joint dataset exceeds max_dataset_bytes')
    digest=hashlib.sha256(json.dumps(digests).encode()+target.contiguous().numpy().tobytes()).hexdigest()
    config={'format':'nil-centralized-awr-v1','agents':specs,'hidden':list(hidden),'lr':lr,'temperature':temperature,
        'max_weight':max_weight,'seed':seed,'batch_size':batch_size,'dataset_sha256':digest}
    model=Model(config);optimizer=torch.optim.Adam(model.parameters(),lr=lr);generator=torch.Generator().manual_seed(seed)
    start=0;history=[]
    if resume:
        payload=checkpoint.read(resume);state=payload['state']
        if state.get('centralized_config')!=config:raise ValueError('Centralized resume requires identical aligned data/returns/config')
        start=state['updates']
        if type(start) is not int or not 0<=start<steps:raise ValueError('Checkpoint already reached requested updates')
        checkpoint.restore(payload,model,optimizer);generator.set_state(state['sampling_rng']);history=list(state['history'])
    joint=torch.cat([record['observations'] for record in records.values()],dim=1)
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True);status='completed';completed=start
    for update in range(start,steps):
        if cancel_event is not None and cancel_event.is_set():status='cancelled';break
        indices=torch.randint(count,(min(batch_size,count),),generator=generator)
        value=model.critic(joint[indices]).squeeze(1);critic_loss=torch.nn.functional.mse_loss(value,target[indices])
        weights=((target[indices]-value.detach())/temperature).clamp(-20,math.log(max_weight)).exp()
        actor_loss=0.
        for actor,record in zip(model.actors,records.values()):
            logits=actor(record['observations'][indices])
            if 'action_masks' in record:logits=logits.masked_fill(~record['action_masks'][indices],-torch.inf)
            actor_loss=actor_loss+(weights*torch.nn.functional.cross_entropy(logits,record['actions'][indices],reduction='none')).mean()
        actor_loss=actor_loss/len(records);loss=critic_loss+actor_loss
        if not torch.isfinite(loss):raise ValueError('Nonfinite centralized AWR objective')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite centralized gradients')
        optimizer.step();completed=update+1
        history.append({'actor_loss':float(actor_loss.detach()),'critic_loss':float(critic_loss.detach())})
    path=checkpoint.save(root/'latest.pt',model,optimizer,centralized_config=config,updates=completed,history=history,sampling_rng=generator.get_state())
    return _result(model,config,history,path,status)


def load(path,*,trusted=False):
    if not trusted:raise ValueError('Centralized local checkpoint loading requires trusted=True')
    payload=checkpoint.read(path);state=payload['state'];config=state.get('centralized_config',{})
    if config.get('format')!='nil-centralized-awr-v1':raise ValueError('Unsupported centralized checkpoint')
    model=Model(config);checkpoint.restore(payload,model)
    return _result(model,config,state['history'],Path(path),'loaded')
