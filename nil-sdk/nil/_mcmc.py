"""Bounded random-walk Metropolis with explicit target identity and exact continuation."""
import copy
import math
from nil.runtime import require
from nil._workflow_io import save as _save,load as _load


class Metropolis:
    def __init__(self,log_density,initial,*,target_id,scale=.5,seed=0):
        np=require('numpy','audio')
        if not callable(log_density) or not isinstance(target_id,str) or not target_id or len(target_id)>1024:raise ValueError('Pure log-density callable and explicit target identity required')
        if type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0 or type(seed) is not int or not 0<=seed<2**32:raise ValueError('Invalid proposal/seed')
        positions=np.asarray(initial,dtype=float)
        if positions.ndim!=2 or not 2<=len(positions)<=32 or not 1<=positions.shape[1]<=256 or not np.isfinite(positions).all():raise ValueError('Finite [2..32 chains,1..256 dimensions] initial states required')
        self.log_density=log_density;self.target_id=target_id;self.scale=scale;self.seed=seed
        self.positions=positions.copy();self.log_prob=self._evaluate(positions);self.rng=np.random.default_rng(seed)
        if not np.isfinite(self.log_prob).all():raise ValueError('Initial chain states need finite target density')
        self.iterations=0;self.accepted=0

    def _evaluate(self,positions):
        np=require('numpy','audio');values=[]
        for row in positions:
            value=float(self.log_density(row.copy()))
            if math.isnan(value) or value==math.inf:raise ValueError('Log density may be finite or negative infinity')
            values.append(value)
        result=np.asarray(values)
        return result

    def sample(self,draws,*,warmup=0,thin=1):
        np=require('numpy','audio')
        if any(type(v) is not int for v in (draws,warmup,thin)) or not 1<=draws<=100000 or not 0<=warmup<=100000 or not 1<=thin<=100 or draws*self.positions.size>1000000 or warmup+draws*thin>1000000:raise ValueError('Sampling budget exceeded')
        if not np.isfinite(self.log_prob).all():raise ValueError('Initial chain states need finite target density')
        # Work on copies: callback failure does not advance sampler/RNG state.
        rng=np.random.default_rng(0);rng.bit_generator.state=copy.deepcopy(self.rng.bit_generator.state)
        positions=self.positions.copy();log_prob=self.log_prob.copy();accepted=0;output=[]
        for step in range(warmup+draws*thin):
            proposal=positions+rng.normal(scale=self.scale,size=positions.shape)
            candidate=self._evaluate(proposal)
            keep=np.log(np.maximum(rng.random(len(positions)),np.finfo(float).tiny))<np.minimum(0,candidate-log_prob)
            positions[keep]=proposal[keep];log_prob[keep]=candidate[keep];accepted+=int(keep.sum())
            if step>=warmup and (step-warmup+1)%thin==0:output.append(positions.copy())
        self.positions=positions;self.log_prob=log_prob;self.rng=rng
        self.iterations+=warmup+draws*thin;self.accepted+=accepted
        values=np.stack(output,axis=1)
        return dict(samples=values,acceptance_rate=accepted/(len(positions)*(warmup+draws*thin)),
                    diagnostics=diagnostics(values),target_id=self.target_id)

    def save(self,path):
        return _save(path,'metropolis',dict(target_id=self.target_id,scale=self.scale,seed=self.seed,
            positions=self.positions.tolist(),log_prob=self.log_prob.tolist(),rng=self.rng.bit_generator.state,
            iterations=self.iterations,accepted=self.accepted))

    @classmethod
    def load(cls,path,*,log_density,target_id):
        np=require('numpy','audio');record=_load(path,'metropolis')
        if record['target_id']!=target_id:raise ValueError('Target identity mismatch')
        sampler=cls(log_density,record['positions'],target_id=target_id,scale=record['scale'],seed=record['seed'])
        if not np.array_equal(sampler.log_prob,np.asarray(record['log_prob'])):raise ValueError('Target density changed at saved chain states')
        if type(record['iterations']) is not int or record['iterations']<0 or type(record['accepted']) is not int or not 0<=record['accepted']<=record['iterations']*len(sampler.positions):raise ValueError('Invalid sampler counters')
        rng=np.random.default_rng(0);rng.bit_generator.state=record['rng']
        sampler.rng=rng;sampler.iterations=record['iterations'];sampler.accepted=record['accepted'];return sampler


def diagnostics(samples):
    """Basic split-Rhat and initial-positive autocorrelation ESS, not rank-normalized."""
    np=require('numpy','audio');values=np.asarray(samples,dtype=float)
    if values.ndim!=3 or values.shape[0]<2 or values.shape[1]<8 or not np.isfinite(values).all():return dict(status='insufficient_draws',split_rhat=None,ess=None)
    chains,draws,dimensions=values.shape;half=draws//2
    split=np.concatenate([values[:,:half],values[:,-half:]],axis=0)
    within=split.var(axis=1,ddof=1).mean(0);between=half*split.mean(1).var(0,ddof=1)
    if (within<=0).any():return dict(status='degenerate_chains',split_rhat=None,ess=None)
    variance=(half-1)/half*within+between/half
    rhat=np.sqrt(variance/within);ess=[]
    for dim in range(dimensions):
        centered=values[:,:,dim]-values[:,:,dim].mean(1,keepdims=True)
        n=1<<(2*draws-1).bit_length();fourier=np.fft.rfft(centered,n=n,axis=1)
        autocov=np.fft.irfft(fourier*fourier.conj(),n=n,axis=1)[:,:draws]/np.arange(draws,0,-1)
        rho=1-(values[:,:,dim].var(1,ddof=1).mean()-autocov.mean(0))/variance[dim]
        total=0.;previous=math.inf
        for lag in range(1,draws-1,2):
            pair=min(previous,float(rho[lag]+rho[lag+1]))
            if pair<=0:break
            total+=pair;previous=pair
        ess.append(min(chains*draws,float(chains*draws/max(1.,1+2*total))))
    return dict(status='estimated',split_rhat=rhat.tolist(),ess=ess,
                method='basic_split_rhat_and_monotone_initial_positive_acf; diagnostic only, not proof of convergence')
