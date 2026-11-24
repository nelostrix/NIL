"""Dependency-free float64 CPU reference tensors/autodiff; no numerical backend imports."""
import json
import math
from pathlib import Path
import random

_MAX_ELEMENTS=100000


def _matrix(data):
    if type(data) in (int,float): data=[[data]]
    if not isinstance(data,(list,tuple)) or not data: raise ValueError('Use a nonempty scalar/list/tuple, not a sparse/external tensor')
    if type(data[0]) in (int,float): data=[data]
    if not all(isinstance(row,(list,tuple)) for row in data): raise ValueError('Tensor requires a rectangular 2D array')
    columns=len(data[0])
    if not columns or len(data)*columns>_MAX_ELEMENTS or any(len(row)!=columns for row in data): raise ValueError('Invalid shape or 100,000-element reference limit exceeded')
    if any(type(value) not in (int,float) or not math.isfinite(value) for row in data for value in row): raise ValueError('Tensor values must be finite numbers')
    return tuple(tuple(float(value) for value in row) for row in data)


def _lists(value): return [_lists(child) for child in value] if isinstance(value,(tuple,list)) else value
def _zeros(shape): return _lists(nest([0.]*math.prod(shape),shape))
def _sigmoid(value): return 1/(1+math.exp(-value)) if value>=0 else math.exp(value)/(1+math.exp(value))


from ._native_tensor import Tensor, nest, same_shape_values, unpack


def tensor(data,**options): return Tensor(data,**options)
def mse(prediction,target): return ((prediction-target)**2).mean()
def binary_cross_entropy(logits,target):
    target=target if isinstance(target,Tensor) else Tensor(same_shape_values(target,logits.shape),shape=logits.shape)
    if target.requires_grad:raise ValueError('BCE targets are constant in the native reference contract')
    if target.shape!=logits.shape or any(not 0<=value<=1 for value in target._flat):raise ValueError('BCE targets must match logits and lie in [0,1]')
    count=logits.numel();z,y=logits._flat,target._flat
    value=sum(max(v,0)+math.log1p(math.exp(-abs(v)))-label*v for v,label in zip(z,y))/count
    shape=(1,1) if logits.ndim==2 else ()
    return Tensor._op_flat([value],shape,(logits,),lambda g:[(logits,[(_sigmoid(v)-label)*g[0]/count for v,label in zip(z,y)])])


def cross_entropy(logits,targets):
    """Mean categorical loss over the last axis, with shape-matched integer labels."""
    if not isinstance(logits,Tensor) or not logits.ndim:raise ValueError('Logits require a class dimension')
    classes=logits.shape[-1];labels=[]
    def collect(value,shape):
        if not shape:
            if type(value) is not int or not 0<=value<classes:raise ValueError('Provide integer class indices within bounds')
            labels.append(value);return
        if not isinstance(value,(list,tuple)) or len(value)!=shape[0]:raise ValueError('Provide one integer class index per logit row')
        for child in value:collect(child,shape[1:])
    collect(targets,logits.shape[:-1]);labels=tuple(labels);rows=len(labels)
    probabilities=logits.log_softmax()
    value=-sum(probabilities._flat[i*classes+label] for i,label in enumerate(labels))/rows
    def backward(g):
        gradient=[0.]*logits.numel()
        for i,label in enumerate(labels):gradient[i*classes+label]=-g[0]/rows
        return [(probabilities,gradient)]
    return Tensor._op_flat([value],(1,1) if logits.ndim==2 else (), (probabilities,),backward)


class Linear:
    def __init__(self,inputs,outputs,*,seed=0):
        if type(inputs) is not int or type(outputs) is not int or min(inputs,outputs)<1 or inputs*outputs>_MAX_ELEMENTS: raise ValueError('Positive integer dimensions required')
        rng=random.Random(seed); scale=1/math.sqrt(inputs)
        self.weight=Tensor([[rng.uniform(-scale,scale) for _ in range(outputs)] for _ in range(inputs)],requires_grad=True)
        self.bias=Tensor([[0.0]*outputs],requires_grad=True)
    def __call__(self,x): return x@self.weight+(self.bias.reshape(self.bias.shape[1]) if x.ndim==1 else self.bias)
    def parameters(self): return [self.weight,self.bias]
    def state_dict(self): return {'format':'nil-native-linear-v1','dtype':'float64','device':'cpu','weight':self.weight.tolist(),'bias':self.bias.tolist()}
    def load_state_dict(self,state):
        if state.get('format')!='nil-native-linear-v1' or state.get('dtype')!='float64' or state.get('device')!='cpu': raise ValueError('Unsupported native model format/dtype/device')
        weights,bias=_matrix(state['weight']),_matrix(state['bias'])
        if (len(weights),len(weights[0]))!=self.weight.shape or (len(bias),len(bias[0]))!=self.bias.shape: raise ValueError('Model shape differs')
        self.weight.assign(weights); self.bias.assign(bias)


class SGD:
    def __init__(self,parameters,*,lr=.01,momentum=0,weight_decay=0):
        self.parameters=list(parameters)
        if not self.parameters or len({id(p) for p in self.parameters})!=len(self.parameters) or any(not isinstance(p,Tensor) or not p.requires_grad or p._parents for p in self.parameters): raise ValueError('Optimizer needs unique differentiable leaf tensors')
        if not all(type(v) in (int,float) and math.isfinite(v) for v in (lr,momentum,weight_decay)) or lr<=0 or not 0<=momentum<1 or weight_decay<0: raise ValueError('Invalid optimizer configuration')
        self.lr,self.momentum,self.weight_decay=lr,momentum,weight_decay; self.steps=0
        self.velocity=[_zeros(p.shape) for p in self.parameters]
    def zero_grad(self):
        for parameter in self.parameters: parameter.zero_grad()
    def step(self):
        staged=[]
        for p,velocity in zip(self.parameters,self.velocity):
            if p.grad is None: raise ValueError('All optimized parameters need gradients')
            gradient=same_shape_values(p.grad,p.shape);previous=same_shape_values(velocity,p.shape)
            v=[self.momentum*old+g+self.weight_decay*value for old,g,value in zip(previous,gradient,p._flat)]
            values=[value-self.lr*moment for value,moment in zip(p._flat,v)]
            unpack(v);unpack(values)
            staged.append((p,nest(values,p.shape),_lists(nest(v,p.shape))))
        for index,(p,values,v) in enumerate(staged): p.assign(values);self.velocity[index]=v
        self.steps+=1
    def state_dict(self): return {'format':'nil-native-sgd-v1','lr':self.lr,'momentum':self.momentum,'weight_decay':self.weight_decay,'steps':self.steps,'velocity':_lists(self.velocity)}
    def load_state_dict(self,state):
        if not isinstance(state,dict) or any(state.get(key)!=getattr(self,key) for key in ('lr','momentum','weight_decay')) or state.get('format')!='nil-native-sgd-v1' or type(state.get('steps')) is not int or state['steps']<0 or not isinstance(state.get('velocity'),list) or len(state['velocity'])!=len(self.parameters): raise ValueError('Optimizer state/config differs')
        values=[_lists(nest(same_shape_values(v,p.shape),p.shape)) for v,p in zip(state['velocity'],self.parameters)]
        self.velocity=values;self.steps=state['steps']



class Adam:
    """Float64 CPU Adam with coupled L2 decay and transactional parameter updates."""
    def __init__(self, parameters, *, lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=0):
        self.parameters=list(parameters)
        if not self.parameters or len({id(p) for p in self.parameters})!=len(self.parameters) or any(not isinstance(p,Tensor) or not p.requires_grad or p._parents for p in self.parameters):
            raise ValueError('Optimizer needs unique differentiable leaf tensors')
        if not isinstance(betas,(list,tuple)) or len(betas)!=2:
            raise ValueError('Provide two Adam decay coefficients')
        if not all(type(value) in (int,float) and math.isfinite(value) for value in (lr,eps,weight_decay,*betas)) or lr<=0 or eps<=0 or weight_decay<0 or any(not 0<=beta<1 for beta in betas):
            raise ValueError('Invalid Adam configuration')
        self.lr,self.betas,self.eps,self.weight_decay=lr,tuple(betas),eps,weight_decay
        self.steps=0
        self.first=[_zeros(p.shape) for p in self.parameters]
        self.second=[_zeros(p.shape) for p in self.parameters]

    def zero_grad(self):
        for parameter in self.parameters:parameter.zero_grad()

    def step(self):
        staged=[];next_step=self.steps+1;b1,b2=self.betas
        for p,first,second in zip(self.parameters,self.first,self.second):
            if p.grad is None:raise ValueError('All optimized parameters need gradients')
            gradient=same_shape_values(p.grad,p.shape)
            previous=same_shape_values(first,p.shape);variances=same_shape_values(second,p.shape)
            moment=[];variance=[];values=[]
            for value,g,old_m,old_v in zip(p._flat,gradient,previous,variances):
                g+=self.weight_decay*value
                m=b1*old_m+(1-b1)*g;v=b2*old_v+(1-b2)*g*g
                moment.append(m);variance.append(v)
                values.append(value-self.lr*(m/(1-b1**next_step))/(math.sqrt(v/(1-b2**next_step))+self.eps))
            unpack(moment);unpack(variance);unpack(values)
            staged.append((p,nest(values,p.shape),_lists(nest(moment,p.shape)),_lists(nest(variance,p.shape))))
        for index,(p,values,first,second) in enumerate(staged):
            p.assign(values);self.first[index]=first;self.second[index]=second
        self.steps=next_step

    def state_dict(self):
        return {'format':'nil-native-adam-v1','lr':self.lr,'betas':list(self.betas),'eps':self.eps,
                'weight_decay':self.weight_decay,'steps':self.steps,'first':_lists(self.first),'second':_lists(self.second)}

    def load_state_dict(self, state):
        if not isinstance(state,dict) or state.get('format')!='nil-native-adam-v1' or any(state.get(key)!=getattr(self,key) for key in ('lr','eps','weight_decay')) or state.get('betas')!=list(self.betas) or type(state.get('steps')) is not int or state['steps']<0:
            raise ValueError('Adam state/config differs')
        buffers=[]
        for key in ('first','second'):
            values=state.get(key)
            if not isinstance(values,list) or len(values)!=len(self.parameters):raise ValueError('Adam state parameter count differs')
            matrices=[same_shape_values(value,p.shape) for value,p in zip(values,self.parameters)]
            if key=='second' and any(value<0 for matrix in matrices for value in matrix):raise ValueError('Adam variance must be nonnegative')
            buffers.append([_lists(nest(value,p.shape)) for value,p in zip(matrices,self.parameters)])
        self.first,self.second=buffers;self.steps=state['steps']


def fit(model,features,targets,*,epochs=100,lr=.01,loss='mse',optimizer='sgd'):
    if type(epochs) is not int or not 1<=epochs<=10000 or loss not in ('mse','bce_logits','cross_entropy') or optimizer not in ('sgd','adam'): raise ValueError('Choose 1–10000 epochs, mse/bce_logits/cross_entropy and sgd/adam')
    x=Tensor(features);y=targets if loss=='cross_entropy' else Tensor(targets)
    optimizer=(SGD if optimizer=='sgd' else Adam)(model.parameters(),lr=lr); history=[]
    for _ in range(epochs):
        optimizer.zero_grad(); prediction=model(x)
        if loss!='cross_entropy' and prediction.shape!=y.shape: raise ValueError('Target shape differs; broadcasting targets is disabled')
        value=cross_entropy(prediction,y) if loss=='cross_entropy' else mse(prediction,y) if loss=='mse' else binary_cross_entropy(prediction,y)
        value.backward(); optimizer.step(); history.append(value.item())
    return {'model':model,'history':history,'steps':epochs}


def save(model,path,*,optimizer=None):
    if optimizer is not None and [id(p) for p in optimizer.parameters]!=[id(p) for p in model.parameters()]:
        raise ValueError('Optimizer parameters must match the saved model in order')
    payload={'format':'nil-native-v1','model':model.state_dict(),'optimizer':optimizer.state_dict() if optimizer else None}
    text=json.dumps(payload,allow_nan=False)
    if len(text.encode())>16000000: raise ValueError('Native artifact exceeds 16 MB')
    with Path(path).open('x',encoding='utf8') as stream: stream.write(text)
    return Path(path)


def load(path):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>16000000: raise ValueError('Invalid native artifact')
    state=json.loads(path.read_text())
    if state.get('format')!='nil-native-v1': raise ValueError('Unsupported native artifact')
    model=_from_state(state['model'])
    optimizer=None
    if state.get('optimizer') is not None:
        config=state['optimizer']
        if config.get('format')=='nil-native-adam-v1':
            optimizer=Adam(model.parameters(),lr=config['lr'],betas=config['betas'],eps=config['eps'],weight_decay=config['weight_decay'])
        elif config.get('format')=='nil-native-sgd-v1':
            optimizer=SGD(model.parameters(),lr=config['lr'],momentum=config['momentum'],weight_decay=config['weight_decay'])
        else:raise ValueError('Unsupported native optimizer format')
        optimizer.load_state_dict(config)
    return model,optimizer


class Tanh:
    def __call__(self,values): return values.tanh()
    def parameters(self): return []
    def state_dict(self): return {'format':'nil-native-tanh-v1'}


class ReLU:
    def __call__(self,values): return values.relu()
    def parameters(self): return []
    def state_dict(self): return {'format':'nil-native-relu-v1'}


class Sequential:
    def __init__(self,*layers):
        if not layers or len(layers)>16 or any(type(layer) not in (Linear,Tanh,ReLU) for layer in layers): raise ValueError('Use 1–16 native Linear/Tanh/ReLU layers')
        if sum(p.shape[0]*p.shape[1] for layer in layers for p in layer.parameters())>_MAX_ELEMENTS: raise ValueError('Network exceeds native parameter budget')
        self.layers=tuple(layers)
    def __call__(self,values):
        for layer in self.layers: values=layer(values)
        return values
    def parameters(self): return [parameter for layer in self.layers for parameter in layer.parameters()]
    def state_dict(self): return {'format':'nil-native-sequential-v1','layers':[layer.state_dict() for layer in self.layers]}
    def load_state_dict(self,state):
        restored=_from_state(state)
        if type(restored) is not Sequential or [type(layer) for layer in restored.layers]!=[type(layer) for layer in self.layers] or [p.shape for p in restored.parameters()]!=[p.shape for p in self.parameters()]: raise ValueError('Native network structure differs')
        for parameter,saved in zip(self.parameters(),restored.parameters()): parameter.assign(saved.data)


def _from_state(state):
    if state.get('format')=='nil-native-linear-v1':
        weights=_matrix(state['weight']); model=Linear(len(weights),len(weights[0])); model.load_state_dict(state); return model
    if state.get('format')=='nil-native-tanh-v1': return Tanh()
    if state.get('format')=='nil-native-relu-v1': return ReLU()
    if state.get('format')=='nil-native-sequential-v1':
        layers=state.get('layers',[])
        if not isinstance(layers,list) or not 0<len(layers)<=16 or any(layer.get('format')=='nil-native-sequential-v1' for layer in layers): raise ValueError('Invalid native layer list')
        return Sequential(*[_from_state(layer) for layer in layers])
    raise ValueError('Unknown native model structure')


def save_tensor(tensor,path):
    if not isinstance(tensor,Tensor):raise TypeError('Expected a native Tensor')
    state={'format':'nil-native-tensor-v1','shape':list(tensor.shape),'dtype':tensor.dtype,'device':tensor.device,
           'requires_grad':tensor.requires_grad,'data':tensor.tolist()}
    text=json.dumps(state,allow_nan=False)
    if len(text.encode())>16000000:raise ValueError('Tensor artifact exceeds 16 MB')
    with Path(path).open('x',encoding='utf8') as stream:stream.write(text)
    return Path(path)


def load_tensor(path):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>16000000:raise ValueError('Invalid tensor artifact')
    state=json.loads(path.read_text())
    if not isinstance(state,dict) or set(state)!={'format','shape','dtype','device','requires_grad','data'} or state['format']!='nil-native-tensor-v1':
        raise ValueError('Unsupported native tensor artifact')
    return Tensor(state['data'],shape=state['shape'],dtype=state['dtype'],device=state['device'],requires_grad=state['requires_grad'])

# Shallow public surface; implementations stay private and dependency-free.
from ._native_sparse import SparseCOO, SparseCSR, save_sparse, load_sparse


def build_cpu_kernel(directory):
    """Explicit local build; requires a compiler/headers, never downloads packages."""
    from ._native_cpu import build_cpu_kernel as build
    return build(directory)


def load_cpu_kernel(path):
    """Load trusted local native code only when explicitly requested."""
    from ._native_cpu import load_cpu_kernel as load
    return load(path)
