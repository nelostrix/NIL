"""Bounded neural behavior cloning and discrete fitted-Q baseline, optional Torch only."""
from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

from nil.runtime import require
from nil._classical_common import positive_int
from nil.training import checkpoint
from nil.lifecycle import sha256

torch=require('torch','torch')


class _PolicyNetwork(torch.nn.Sequential):
    def set_training_mode(self,mode): self.train(mode)


class OfflinePolicy:
    _nil_rl_policy=True
    def __init__(self,config):
        self.config=config; self.device=torch.device('cpu')
        dimensions=[config['input_dim'],*config['hidden'],config['output_dim']]
        if len(dimensions)>10 or any(type(v) is not int or v<1 or v>4096 for v in dimensions) or sum((a+1)*b for a,b in zip(dimensions,dimensions[1:]))>1000000: raise ValueError('Offline network exceeds supported shape/parameter budget')
        layers=[]
        with torch.random.fork_rng():
            torch.manual_seed(config['seed'])
            for index,(left,right) in enumerate(zip(dimensions,dimensions[1:])):
                layers.append(torch.nn.Linear(left,right))
                if index<len(dimensions)-2: layers.append(torch.nn.Tanh())
        self.policy=_PolicyNetwork(*layers).eval(); self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=config['lr'])
        self.target=copy.deepcopy(self.policy); self.generator=torch.Generator().manual_seed(config['seed'])
        self.action_generator=torch.Generator().manual_seed(config['seed']+1); self.state={}
    def get_env(self): return None
    def get_vec_normalize_env(self): return None
    def predict(self,observation,*,deterministic=True,action_masks=None):
        values=torch.as_tensor(observation,dtype=torch.float32); single=values.ndim==1
        if single: values=values[None,:]
        if values.ndim!=2 or values.shape[1]!=self.config['input_dim'] or not torch.isfinite(values).all(): raise ValueError('Expected finite flat observation vectors')
        previous=self.policy.training; self.policy.eval()
        try:
            with torch.inference_mode():
                output=self.policy(values)
                if not torch.isfinite(output).all(): raise ValueError('Nonfinite offline policy output')
                if self.config['continuous']:
                    if action_masks is not None: raise ValueError('Continuous policies do not accept discrete masks')
                    bounds=self.config.get('action_bounds')
                    if bounds is not None: output=torch.maximum(torch.minimum(output,torch.tensor(bounds[1])),torch.tensor(bounds[0]))
                    action=output.numpy()
                else:
                    if action_masks is not None:
                        mask=torch.as_tensor(action_masks,dtype=torch.bool)
                        if single and mask.ndim==1: mask=mask[None,:]
                        if mask.shape!=output.shape or not mask.any(dim=1).all(): raise ValueError('Invalid policy action mask')
                        output=output.masked_fill(~mask,-torch.inf)
                    action=(output.argmax(dim=1) if deterministic else torch.multinomial(output.softmax(dim=1),1,generator=self.action_generator).squeeze(1)).numpy()
                return (action[0] if single else action),None
        finally: self.policy.train(previous)
    def save(self,path): return save(self,path)


@dataclass
class OfflineResult:
    model: OfflinePolicy
    history: list
    checkpoint: Path
    status: str
    _nil_rl_result=True
    def predict(self,*args,**kwargs): return self.model.predict(*args,**kwargs)


def _dataset(data,algorithm,action_count,max_bytes):
    if not isinstance(data,dict): raise TypeError('Offline data must be an explicit array mapping')
    required={'observations','actions'} if algorithm=='behavior_cloning' else {'observations','actions','next_observations','rewards','terminated','truncated'}
    if not required<=set(data) or set(data)-required-{'action_masks','next_action_masks'}: raise ValueError('Offline dataset fields differ from the algorithm schema')
    values={name:torch.as_tensor(value).detach().cpu() for name,value in data.items()}
    if sum(value.numel()*value.element_size() for value in values.values())>max_bytes: raise ValueError('Offline dataset exceeds max_dataset_bytes')
    x=values['observations']; actions=values['actions']
    # Account for normalized tensor dtypes before float32/int64 copies allocate.
    normalized_bytes=sum(value.numel()*(8 if name=='actions' and value.ndim==1 else 1 if name in ('action_masks','next_action_masks','terminated','truncated') else 4) for name,value in values.items())
    if normalized_bytes>max_bytes: raise ValueError('Normalized offline dataset exceeds max_dataset_bytes')
    if x.ndim!=2: raise ValueError('Observations must be a finite rows×features matrix')
    count=len(x)
    if x.ndim!=2 or not count or x.shape[1]>4096 or any(value.ndim<1 or value.is_complex() or not torch.isfinite(value).all() or len(value)!=count for value in values.values()): raise ValueError('Offline arrays require matching finite rows')
    continuous=actions.ndim==2
    if continuous:
        if algorithm!='behavior_cloning' or not actions.shape[1]: raise ValueError('Offline fitted Q supports discrete actions only')
        outputs=actions.shape[1]; values['actions']=actions.float()
    else:
        positive_int(action_count,'action_count'); outputs=action_count
        if actions.ndim!=1 or (actions<0).any() or not torch.equal(actions,actions.round()) or (actions>=outputs).any(): raise ValueError('Discrete actions must be integral and within action_count')
        values['actions']=actions.long()
    values['observations']=x.float()
    for name in ('action_masks','next_action_masks'):
        if name in values:
            original=values[name]
            if continuous or original.shape!=(count,outputs) or not ((original==0)|(original==1)).all(): raise ValueError('Action masks must be binary rows×action_count')
            values[name]=original.bool()
    if 'action_masks' in values:
        mask=values['action_masks']
        if not mask.any(dim=1).all() or not mask.gather(1,values['actions'][:,None]).all(): raise ValueError('Dataset contains masked-out actions')
    if algorithm=='offline_dqn':
        if values['next_observations'].shape!=x.shape or values['rewards'].shape!=(count,): raise ValueError('Transition observation/reward shape differs')
        for name in ('terminated','truncated'):
            if values[name].shape!=(count,) or not ((values[name]==0)|(values[name]==1)).all(): raise ValueError('Terminal/truncation flags must be binary vectors')
            values[name]=values[name].bool()
        values['next_observations']=values['next_observations'].float(); values['rewards']=values['rewards'].float()
        if 'next_action_masks' in values and not (values['next_action_masks'].any(dim=1)|values['terminated']).all(): raise ValueError('Nonterminal transitions need a valid next action, including truncations')
    digest=hashlib.sha256()
    for name,value in sorted(values.items()):
        digest.update(json.dumps([name,list(value.shape),str(value.dtype)]).encode()); digest.update(memoryview(value.contiguous().numpy()))
    return values,continuous,outputs,digest.hexdigest()


def fit(data,*,algorithm='behavior_cloning',action_count=None,steps=500,batch_size=64,
        hidden=(64,64),lr=.001,gamma=.99,target_update=50,seed=0,action_bounds=None,
        max_dataset_bytes=64000000,output_dir='./nil_offline',resume=None,cancel_event=None):
    if algorithm not in ('behavior_cloning','offline_dqn'): raise ValueError('Choose behavior_cloning or offline_dqn')
    for name,value in [('steps',steps),('batch_size',batch_size),('target_update',target_update),('max_dataset_bytes',max_dataset_bytes)]: positive_int(value,name)
    if type(seed) is not int or not 0<=seed<2**32 or not math.isfinite(lr) or lr<=0 or not math.isfinite(gamma) or not 0<=gamma<=1: raise ValueError('Invalid offline optimizer/seed/discount')
    values,continuous,outputs,digest=_dataset(data,algorithm,action_count,max_dataset_bytes)
    if action_bounds is not None:
        low,high=[torch.as_tensor(item,dtype=torch.float32) for item in action_bounds]
        if not continuous or low.shape!=(outputs,) or high.shape!=low.shape or not torch.isfinite(low).all() or not torch.isfinite(high).all() or not (low<high).all(): raise ValueError('Provide finite ordered action bounds matching continuous outputs')
        if ((values['actions']<low)|(values['actions']>high)).any(): raise ValueError('Dataset actions exceed supplied bounds')
        action_bounds=[low.tolist(),high.tolist()]
    config={'algorithm':algorithm,'input_dim':values['observations'].shape[1],'output_dim':outputs,'continuous':continuous,'hidden':list(hidden),'lr':lr,'gamma':gamma,'target_update':target_update,'seed':seed,'batch_size':batch_size,'dataset_sha256':digest,'action_bounds':action_bounds}
    policy=OfflinePolicy(config); start=0; history=[]
    if resume:
        payload=checkpoint.read(resume)
        if payload['state'].get('offline_config')!=config: raise ValueError('Offline resume requires the same dataset/config')
        state=payload['state']
        if state['updates']>=steps: raise ValueError('Checkpoint already reached requested total updates')
        policy.generator.set_state(state['sampling_rng']); policy.action_generator.set_state(state['action_rng'])
        policy.target.load_state_dict(state['target']); checkpoint.restore(payload,policy.policy,policy.optimizer)
        start,history=state['updates'],state['history']
    root=Path(output_dir); root.mkdir(parents=True,exist_ok=True); status='completed'; completed=start
    for update in range(start,steps):
        if cancel_event is not None and cancel_event.is_set(): status='cancelled'; break
        indices=torch.randint(len(values['observations']),(min(batch_size,len(values['observations'])),),generator=policy.generator)
        x=values['observations'][indices]; actions=values['actions'][indices]; output=policy.policy(x)
        if algorithm=='behavior_cloning':
            if continuous: loss=torch.nn.functional.mse_loss(output,actions)
            else:
                if 'action_masks' in values: output=output.masked_fill(~values['action_masks'][indices],-torch.inf)
                loss=torch.nn.functional.cross_entropy(output,actions)
        else:
            with torch.no_grad():
                future=policy.target(values['next_observations'][indices])
                if 'next_action_masks' in values: future=future.masked_fill(~values['next_action_masks'][indices],-torch.inf)
                best=future.max(dim=1).values
                best=torch.where(values['terminated'][indices],torch.zeros_like(best),best)
                target=values['rewards'][indices]+gamma*best
            loss=torch.nn.functional.smooth_l1_loss(output.gather(1,actions[:,None]).squeeze(1),target)
        if not torch.isfinite(loss): raise ValueError('Nonfinite offline objective')
        policy.optimizer.zero_grad(set_to_none=True); loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in policy.policy.parameters()): raise ValueError('Nonfinite offline gradients')
        policy.optimizer.step(); completed=update+1
        if algorithm=='offline_dqn' and completed%target_update==0: policy.target.load_state_dict(policy.policy.state_dict())
        history.append(float(loss.detach()))
    policy.policy.eval()
    policy.state={'updates':completed,'history':history,'offline_config':config,'sampling_rng':policy.generator.get_state(),'action_rng':policy.action_generator.get_state(),'target':policy.target.state_dict()}
    artifact=checkpoint.save(root/'latest.pt',policy.policy,policy.optimizer,**policy.state)
    return OfflineResult(policy,history,artifact,status)


def save(policy,path):
    destination=Path(path)
    if destination.exists() or destination.is_symlink(): raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.nil-offline-',dir=destination.parent) as temporary:
        root=Path(temporary)
        state=dict(policy.state,sampling_rng=policy.generator.get_state(),action_rng=policy.action_generator.get_state(),target=policy.target.state_dict())
        checkpoint.save(root/'model.pt',policy.policy,policy.optimizer,**state)
        manifest={'format':'nil-rl-offline-v1','files':{part.name:sha256(part) for part in root.iterdir()}}
        (root/'manifest.json').write_text(json.dumps(manifest)); destination.mkdir()
        try:
            for part in root.iterdir(): part.replace(destination/part.name)
        except BaseException: shutil.rmtree(destination); raise
    return destination


def load(path,*,trusted=False):
    if not trusted: raise ValueError('Loading a local policy requires trusted=True')
    root=Path(path)
    if root.is_symlink() or (root/'manifest.json').is_symlink(): raise ValueError('Policy paths must not be symlinks')
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('format')!='nil-rl-offline-v1' or not isinstance(manifest.get('files'),dict) or not {'model.pt','model.pt.json'}<=set(manifest['files']): raise ValueError('Invalid offline policy format')
    for name,digest in manifest['files'].items():
        part=root/name
        if Path(name).name!=name or part.is_symlink() or sha256(part)!=digest: raise ValueError('Policy integrity validation failed')
    payload=checkpoint.read(root/'model.pt'); state=payload['state']; policy=OfflinePolicy(state['offline_config'])
    checkpoint.restore(payload,policy.policy,policy.optimizer); policy.target.load_state_dict(state['target'])
    policy.generator.set_state(state['sampling_rng']); policy.action_generator.set_state(state['action_rng']); policy.state=state
    return policy
