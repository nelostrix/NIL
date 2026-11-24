"""Conjugate probability models and explicit cross-fitted causal ATE estimation."""
import math
from statistics import NormalDist
from nil._workflow_io import save as _save, load as _load
from nil.runtime import require


class BetaBernoulli:
    """Beta prior and binary observations; exact posterior mean/variance."""
    def __init__(self, alpha=1., beta=1.):
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<v<=1e12 for v in (alpha,beta)):
            raise ValueError('Prior parameters must be positive finite numbers')
        self.alpha=float(alpha);self.beta=float(beta)

    def update(self, observations):
        yes=no=0
        for i,value in enumerate(observations):
            if i>=1000000:raise ValueError('Update exceeds one million observations')
            if type(value) not in (bool,int) or value not in (0,1):raise ValueError('Expected binary observations')
            yes+=int(value);no+=1-int(value)
        if max(self.alpha+yes,self.beta+no)>1e12:raise ValueError('Posterior count budget exceeded')
        self.alpha+=yes;self.beta+=no;return self

    def predict(self):
        total=self.alpha+self.beta
        probability=self.alpha/total
        return dict(probability=probability,variance=probability*(self.beta/total)/(total+1))

    def save(self,path):return _save(path,'beta-bernoulli',dict(alpha=self.alpha,beta=self.beta))

    @classmethod
    def load(cls,path):return cls(**_load(path,'beta-bernoulli'))


class BayesianLinear:
    """Gaussian linear likelihood, fixed known noise, isotropic Gaussian prior.

    No automatic intercept: add a constant feature explicitly. Predictive standard
    deviation includes observation noise. Posterior operations use NumPy.
    """
    def __init__(self, *, prior_precision=1., noise_variance=1.):
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (prior_precision,noise_variance)):
            raise ValueError('Precisions/variances must be positive finite numbers')
        self.prior_precision=prior_precision;self.noise_variance=noise_variance

    def fit(self, features, targets):
        np=require('numpy','classical');x=np.asarray(features,dtype=float);y=np.asarray(targets,dtype=float)
        if x.ndim!=2 or not 1<=x.shape[1]<=512 or not 1<=len(x)<=100000 or y.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Finite bounded [N,F] features and [N] targets required')
        precision=self.prior_precision*np.eye(x.shape[1])+x.T@x/self.noise_variance
        cholesky=np.linalg.cholesky(precision);mean=np.linalg.solve(precision,x.T@y/self.noise_variance)
        if not np.isfinite(cholesky).all() or not np.isfinite(mean).all():raise ValueError('Nonfinite posterior')
        self.cholesky_=cholesky;self.mean_=mean
        return self

    def predict(self, features):
        np=require('numpy','classical')
        if not hasattr(self,'mean_'):raise ValueError('Fit the posterior first')
        x=np.asarray(features,dtype=float)
        if x.ndim!=2 or x.shape[1]!=len(self.mean_) or len(x)>100000 or not np.isfinite(x).all():raise ValueError('Invalid prediction features')
        projected=np.linalg.solve(self.cholesky_,x.T)
        return dict(mean=x@self.mean_,std=np.sqrt(self.noise_variance+(projected*projected).sum(axis=0)))

    def save(self,path):
        if not hasattr(self,'mean_'):raise ValueError('Fit the posterior first')
        return _save(path,'bayesian-linear',dict(prior_precision=self.prior_precision,noise_variance=self.noise_variance,
                    mean=self.mean_.tolist(),cholesky=self.cholesky_.tolist()))

    @classmethod
    def load(cls,path):
        np=require('numpy','classical');r=_load(path,'bayesian-linear')
        model=cls(prior_precision=r['prior_precision'],noise_variance=r['noise_variance'])
        mean=np.asarray(r['mean'],dtype=float);ch=np.asarray(r['cholesky'],dtype=float)
        if mean.ndim!=1 or not 1<=len(mean)<=512 or ch.shape!=(len(mean),len(mean)) or not np.isfinite(mean).all() or not np.isfinite(ch).all() or not np.array_equal(ch,np.tril(ch)) or (np.diag(ch)<=0).any():raise ValueError('Invalid posterior state')
        model.mean_=mean;model.cholesky_=ch;return model


def causal_ate(features, treatment, outcome, *, folds=3, seed=0, overlap=.01, confidence=.95,
               propensity_model=None, outcome_model=None):
    """Cross-fitted AIPW for binary treatment and finite continuous outcomes.

    Assumes consistency, positivity, independent observations and no unmeasured
    confounding given features. Cannot establish these assumptions from data.
    Fitted propensities outside overlap bounds fail rather than silently clip.
    Interval is asymptotic influence-function normal approximation, not a
    finite-sample or post-model-selection coverage guarantee.
    """
    np=require('numpy','classical');require('sklearn','classical')
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import StratifiedKFold
    if type(folds) is not int or not 2<=folds<=10 or type(seed) is not int or not 0<=seed<2**32:raise ValueError('Invalid fold count or seed')
    if type(overlap) not in (int,float) or not 0<overlap<.5 or type(confidence) not in (int,float) or not 0<confidence<1:raise ValueError('Invalid overlap/confidence')
    x=np.asarray(features,dtype=float);t=np.asarray(treatment);y=np.asarray(outcome,dtype=float)
    if x.ndim!=2 or not 1<=x.shape[1]<=512 or not 20<=len(x)<=100000 or t.shape!=(len(x),) or y.shape!=t.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(t,[0,1]).all():raise ValueError('Finite bounded data and binary treatment required')
    t=t.astype(int)
    if min((t==0).sum(),(t==1).sum())<folds:raise ValueError('Both treatment groups need enough samples per fold')
    propensity=LogisticRegression(max_iter=1000,random_state=seed) if propensity_model is None else propensity_model
    regression=Ridge(alpha=1.) if outcome_model is None else outcome_model
    pseudo=np.empty(len(y));scores=np.empty(len(y));fold_ids=np.empty(len(y),dtype=int)
    for number,(train,test) in enumerate(StratifiedKFold(folds,shuffle=True,random_state=seed).split(x,t)):
        p=clone(propensity).fit(x[train],t[train]);pred=p.predict_proba(x[test])
        if list(p.classes_)!=[0,1] or pred.shape!=(len(test),2):raise ValueError('Propensity estimator must provide binary probabilities')
        if not np.isfinite(pred).all() or (pred<0).any() or (pred>1).any() or not np.allclose(pred.sum(1),1):raise ValueError('Invalid propensity probabilities')
        e=pred[:,1]
        if not np.isfinite(e).all() or (e<overlap).any() or (e>1-overlap).any():raise ValueError('Propensity overlap failed; revise design/model, not hidden clipping')
        m0=clone(regression).fit(x[train[t[train]==0]],y[train[t[train]==0]]).predict(x[test])
        m1=clone(regression).fit(x[train[t[train]==1]],y[train[t[train]==1]]).predict(x[test])
        pseudo[test]=m1-m0+t[test]*(y[test]-m1)/e-(1-t[test])*(y[test]-m0)/(1-e)
        scores[test]=e;fold_ids[test]=number
    if not np.isfinite(pseudo).all():raise ValueError('Nonfinite nuisance estimates')
    estimate=float(pseudo.mean());se=float(pseudo.std(ddof=1)/math.sqrt(len(y)))
    if not math.isfinite(estimate) or not math.isfinite(se):raise ValueError('Nonfinite causal estimate/variance')
    z=NormalDist().inv_cdf((1+confidence)/2)
    return dict(schema_version=1,method='cross_fitted_aipw',ate=estimate,standard_error=se,
                confidence=confidence,interval=[estimate-z*se,estimate+z*se],samples=len(y),folds=folds,
                propensity_range=[float(scores.min()),float(scores.max())],fold_ids=fold_ids.tolist(),
                assumptions=['consistency','positivity','no_unmeasured_confounding','independent_observations'])
