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


def _zeros(shape): return [[0.0]*shape[1] for _ in range(shape[0])]
def _sigmoid(value): return 1/(1+math.exp(-value)) if value>=0 else math.exp(value)/(1+math.exp(value))


class Tensor:
    def __init__(self,data,*,requires_grad=False,dtype='float64',device='cpu'):
        if dtype!='float64' or device!='cpu': raise ValueError('Native reference tensors support float64 CPU only')
        if type(requires_grad) is not bool: raise TypeError('requires_grad must be boolean')
        self.dtype,self.device='float64','cpu'
        self._data=_matrix(data); self.requires_grad=requires_grad; self.grad=None; self._parents=(); self._vjp=None; self._versions=(); self._version=0
    @property
    def data(self): return self._data
    @property
    def shape(self): return len(self.data),len(self.data[0])
    def tolist(self): return [list(row) for row in self.data]
    def item(self):
        if self.shape!=(1,1): raise ValueError('item requires one element')
        return self.data[0][0]
    def assign(self,data):
        values=_matrix(data)
        if (len(values),len(values[0]))!=self.shape: raise ValueError('Assigned shape differs')
        self._data=values; self._version+=1
    def zero_grad(self): self.grad=None
    @classmethod
    def _op(cls,data,parents,vjp):
        result=cls(data,requires_grad=any(parent.requires_grad for parent in parents))
        if result.requires_grad:
            result._parents=parents; result._vjp=vjp; result._versions=tuple(parent._version for parent in parents)
        return result
    def _binary(self,other,operation,derivatives):
        other=other if isinstance(other,Tensor) else Tensor(other)
        ar,ac=self.shape; br,bc=other.shape; rows,cols=max(ar,br),max(ac,bc)
        if ar not in (1,rows) or br not in (1,rows) or ac not in (1,cols) or bc not in (1,cols) or rows*cols>_MAX_ELEMENTS: raise ValueError('Incompatible broadcast shape or size limit')
        a,b=self.data,other.data
        values=[[operation(a[i%ar][j%ac],b[i%br][j%bc]) for j in range(cols)] for i in range(rows)]
        def backward(gradient):
            ga,gb=_zeros(self.shape),_zeros(other.shape)
            for i in range(rows):
                for j in range(cols):
                    da,db=derivatives(a[i%ar][j%ac],b[i%br][j%bc]); g=gradient[i][j]
                    ga[i%ar][j%ac]+=g*da; gb[i%br][j%bc]+=g*db
            return [(self,ga),(other,gb)]
        return Tensor._op(values,(self,other),backward)
    def __add__(self,other): return self._binary(other,lambda a,b:a+b,lambda a,b:(1,1))
    __radd__=__add__
    def __neg__(self): return self*-1
    def __sub__(self,other): return self+-other if isinstance(other,Tensor) else self+(-other)
    def __rsub__(self,other): return -self+other
    def __mul__(self,other): return self._binary(other,lambda a,b:a*b,lambda a,b:(b,a))
    __rmul__=__mul__
    def __truediv__(self,other): return self._binary(other,lambda a,b:a/b,lambda a,b:(1/b,-a/(b*b)))
    def __rtruediv__(self,other): return Tensor(other)/self
    def __pow__(self,power):
        if type(power) not in (int,float) or not math.isfinite(power): raise ValueError('Power must be a finite scalar')
        return self._unary(lambda x:x**power,lambda x:0 if power==0 else power*x**(power-1))
    def _unary(self,function,derivative):
        values=[[function(value) for value in row] for row in self.data]; data=self.data
        return Tensor._op(values,(self,),lambda g:[(self,[[g[i][j]*derivative(data[i][j]) for j in range(self.shape[1])] for i in range(self.shape[0])])])
    def relu(self): return self._unary(lambda x:max(0,x),lambda x:float(x>0))
    def tanh(self): return self._unary(math.tanh,lambda x:1-math.tanh(x)**2)
    def sigmoid(self): return self._unary(_sigmoid,lambda x:_sigmoid(x)*(1-_sigmoid(x)))
    def exp(self): return self._unary(math.exp,math.exp)
    def log(self): return self._unary(math.log,lambda x:1/x)
    def sum(self):
        value=sum(sum(row) for row in self.data)
        return Tensor._op([[value]],(self,),lambda g:[(self,[[g[0][0]]*self.shape[1] for _ in range(self.shape[0])])])
    def mean(self): return self.sum()/(self.shape[0]*self.shape[1])
    def matmul(self,other):
        if not isinstance(other,Tensor) or self.shape[1]!=other.shape[0]: raise ValueError('Matrix multiplication dimensions differ')
        m,k=self.shape; _,n=other.shape
        if m*n>_MAX_ELEMENTS: raise ValueError('Matrix multiplication output exceeds 100,000-element reference limit')
        a,b=self.data,other.data
        values=[[sum(a[i][p]*b[p][j] for p in range(k)) for j in range(n)] for i in range(m)]
        def backward(g):
            return [(self,[[sum(g[i][j]*b[p][j] for j in range(n)) for p in range(k)] for i in range(m)]),
                    (other,[[sum(a[i][p]*g[i][j] for i in range(m)) for j in range(n)] for p in range(k)])]
        return Tensor._op(values,(self,other),backward)
    __matmul__=matmul
    def backward(self):
        if self.shape!=(1,1) or not self.requires_grad: raise ValueError('backward requires a differentiable scalar')
        order=[]; seen=set(); stack=[(self,False)]
        while stack:
            node,visited=stack.pop()
            if visited: order.append(node); continue
            if node in seen: continue
            seen.add(node); stack.append((node,True)); stack.extend((p,False) for p in reversed(node._parents))
        for node in order:
            if tuple(p._version for p in node._parents)!=node._versions: raise RuntimeError('A tensor changed after this graph was constructed')
        gradients={self:[[1.0]]}
        for node in reversed(order):
            if node not in gradients or node._vjp is None: continue
            for parent,values in node._vjp(gradients[node]):
                if not parent.requires_grad: continue
                current=gradients.setdefault(parent,_zeros(parent.shape))
                for i in range(parent.shape[0]):
                    for j in range(parent.shape[1]): current[i][j]+=values[i][j]
        updates=[]
        for node,values in gradients.items():
            if node._parents or not node.requires_grad: continue
            combined=[[value+(node.grad[i][j] if node.grad is not None else 0) for j,value in enumerate(row)] for i,row in enumerate(values)]
            updates.append((node,_matrix(combined)))
        for node,values in updates: node.grad=values


def tensor(data,**options): return Tensor(data,**options)
def mse(prediction,target): return ((prediction-target)**2).mean()
def binary_cross_entropy(logits,target):
    target=target if isinstance(target,Tensor) else Tensor(target)
    if target.requires_grad: raise ValueError('BCE targets are constant in the native reference contract')
    if target.shape!=logits.shape or any(not 0<=value<=1 for row in target.data for value in row): raise ValueError('BCE targets must match logits and lie in [0,1]')
    count=logits.shape[0]*logits.shape[1]; z,y=logits.data,target.data
    value=sum(max(v,0)+math.log1p(math.exp(-abs(v)))-y[i][j]*v for i,row in enumerate(z) for j,v in enumerate(row))/count
    return Tensor._op([[value]],(logits,),lambda g:[(logits,[[( _sigmoid(v)-y[i][j])*g[0][0]/count for j,v in enumerate(row)] for i,row in enumerate(z)])])


class Linear:
    def __init__(self,inputs,outputs,*,seed=0):
        if type(inputs) is not int or type(outputs) is not int or min(inputs,outputs)<1 or inputs*outputs>_MAX_ELEMENTS: raise ValueError('Positive integer dimensions required')
        rng=random.Random(seed); scale=1/math.sqrt(inputs)
        self.weight=Tensor([[rng.uniform(-scale,scale) for _ in range(outputs)] for _ in range(inputs)],requires_grad=True)
        self.bias=Tensor([[0.0]*outputs],requires_grad=True)
    def __call__(self,x): return x@self.weight+self.bias
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
            v=[[self.momentum*velocity[i][j]+p.grad[i][j]+self.weight_decay*p.data[i][j] for j in range(p.shape[1])] for i in range(p.shape[0])]
            new=_matrix([[p.data[i][j]-self.lr*v[i][j] for j in range(p.shape[1])] for i in range(p.shape[0])]); staged.append((p,new,v))
        for index,(p,values,v) in enumerate(staged): p.assign(values); self.velocity[index]=v
        self.steps+=1
    def state_dict(self): return {'format':'nil-native-sgd-v1','lr':self.lr,'momentum':self.momentum,'weight_decay':self.weight_decay,'steps':self.steps,'velocity':self.velocity}
    def load_state_dict(self,state):
        if any(state.get(key)!=getattr(self,key) for key in ('lr','momentum','weight_decay')) or state.get('format')!='nil-native-sgd-v1' or type(state.get('steps')) is not int or state['steps']<0 or len(state.get('velocity',[]))!=len(self.parameters): raise ValueError('Optimizer state/config differs')
        values=[_matrix(v) for v in state['velocity']]
        if any((len(v),len(v[0]))!=p.shape for p,v in zip(self.parameters,values)): raise ValueError('Optimizer state shape differs')
        self.velocity=[[list(row) for row in v] for v in values]; self.steps=state['steps']


def fit(model,features,targets,*,epochs=100,lr=.01,loss='mse'):
    if type(epochs) is not int or not 1<=epochs<=10000 or loss not in ('mse','bce_logits'): raise ValueError('Choose 1–10000 epochs and mse/bce_logits')
    x,y=Tensor(features),Tensor(targets); optimizer=SGD(model.parameters(),lr=lr); history=[]
    for _ in range(epochs):
        optimizer.zero_grad(); prediction=model(x)
        if prediction.shape!=y.shape: raise ValueError('Target shape differs; broadcasting targets is disabled')
        value=mse(prediction,y) if loss=='mse' else binary_cross_entropy(prediction,y)
        value.backward(); optimizer.step(); history.append(value.item())
    return {'model':model,'history':history,'steps':epochs}


def save(model,path,*,optimizer=None):
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
        config=state['optimizer']; optimizer=SGD(model.parameters(),lr=config['lr'],momentum=config['momentum'],weight_decay=config['weight_decay']); optimizer.load_state_dict(config)
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
