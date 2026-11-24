"""Independent posterior/causal references and held-out learning regimes."""
import pytest
np=pytest.importorskip('numpy')
pytest.importorskip('sklearn')
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import nil


def test_bayesian_linear_matches_closed_form_and_reload(tmp_path):
    x=np.array([[1.,-1.],[1.,0.],[1.,1.],[1.,2.]]);y=2*x[:,1]+1
    model=nil.probabilistic.BayesianLinear(prior_precision=.1,noise_variance=.5).fit(x,y)
    covariance=np.linalg.inv(.1*np.eye(2)+x.T@x/.5)
    expected=covariance@x.T@y/.5
    output=model.predict(x)
    np.testing.assert_allclose(output['mean'],x@expected)
    np.testing.assert_allclose(output['std']**2,.5+np.einsum('ij,jk,ik->i',x,covariance,x))
    restored=nil.probabilistic.BayesianLinear.load(model.save(tmp_path/'model'))
    np.testing.assert_array_equal(restored.predict(x)['mean'],output['mean'])


def test_causal_cross_fit_recovers_known_effect_and_does_not_fit_callers():
    rng=np.random.default_rng(7);x=rng.normal(size=(1200,3));p=1/(1+np.exp(-.4*x[:,0]))
    t=rng.binomial(1,p);y=3*t+2*x[:,0]+rng.normal(scale=.3,size=len(x))
    propensity=LogisticRegression(max_iter=1000)
    r=nil.probabilistic.causal_ate(x,t,y,propensity_model=propensity)
    assert abs(r['ate']-3)<.12 and r['interval'][0]<3<r['interval'][1]
    assert set(r['fold_ids'])=={0,1,2} and not hasattr(propensity,'classes_')
    assert r==nil.probabilistic.causal_ate(x,t,y)
    with pytest.raises(ValueError,match='groups'):nil.probabilistic.causal_ate(x,np.zeros(len(x)),y)
    with pytest.raises(ValueError,match='overlap'):nil.probabilistic.causal_ate(x,t,y,overlap=.49)


def test_self_training_real_iris_and_model_persistence(tmp_path):
    x,y=load_iris(return_X_y=True)
    train,test=train_test_split(np.arange(len(y)),stratify=y,random_state=7,test_size=.3)
    labels=y[train].copy();labels[::2]=-1
    estimator=LogisticRegression(max_iter=1000)
    model=nil.learning.self_training(estimator,x[train],labels,threshold=.8)
    assert (model.predict(x[test])==y[test]).mean()>.85
    assert not hasattr(estimator,'classes_')
    artifact=nil.save(model,tmp_path/'model')
    restored=nil.load(artifact,trusted=True)
    np.testing.assert_array_equal(nil.predict(restored,x[test]),model.predict(x[test]))


def test_active_learning_budget_and_few_shot_support():
    x,y=load_iris(return_X_y=True)
    indices=np.array([0,5,50,55,100,105]);model=LogisticRegression(max_iter=1000).fit(x[indices],y[indices])
    selected=nil.learning.query_uncertain(model,x,budget=8,excluded=indices.tolist())
    assert len(set(selected))==8 and not set(selected)&set(indices)
    active=LogisticRegression(max_iter=1000).fit(x[np.r_[indices,selected]],y[np.r_[indices,selected]])
    probabilities=model.predict_proba(x)
    entropy=-(probabilities*np.log(np.maximum(probabilities,1e-300))).sum(1)
    expected=sorted((i for i in range(len(x)) if i not in indices),key=lambda i:(-entropy[i],i))[:8]
    assert selected==expected
    assert np.isfinite(active.coef_).all() and not np.array_equal(active.coef_,model.coef_)
    r=nil.learning.prototypes([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]],[0,0,1,1],[[-3.,0.],[3.,0.]])
    assert r['labels'].tolist()==[0,1]
    with pytest.raises(ValueError):nil.learning.query_uncertain(model,x,budget=1000)


def test_replay_exact_restart_and_owned_records(tmp_path):
    memory=nil.learning.ReplayMemory(10,seed=7)
    for i in range(30):memory.add(dict(value=i))
    restored=nil.learning.ReplayMemory.load(memory.save(tmp_path/'replay'))
    for i in range(30,100):memory.add(dict(value=i));restored.add(dict(value=i))
    assert len(memory.records())==10 and memory.records()==restored.records()
    result=memory.records();result.clear();assert len(memory.records())==10
    before=memory.records()
    with pytest.raises(ValueError):memory.add(dict(value=float('nan')))
    assert memory.records()==before and memory.seen==100
