"""Real Iris semi/active/few-shot learning and simulated identified causal ATE."""
import argparse
import json
from pathlib import Path


def exercise(output):
    import nil
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    x,y=load_iris(return_X_y=True)
    train,test=train_test_split(np.arange(len(y)),stratify=y,random_state=7,test_size=.3)
    labels=y[train].copy();labels[::2]=-1
    estimator=LogisticRegression(max_iter=1000)
    semi=nil.learning.self_training(estimator,x[train],labels,threshold=.8)
    semi_score=float((semi.predict(x[test])==y[test]).mean())
    if semi_score<.8:raise AssertionError('Declared held-out Iris quality floor failed')
    restored=nil.load(nil.save(semi,output/'semi-model'),trusted=True)
    np.testing.assert_array_equal(restored.predict(x[test]),semi.predict(x[test]))
    initial=np.concatenate([np.flatnonzero(y[train]==c)[:8] for c in range(3)])
    active=LogisticRegression(max_iter=1000).fit(x[train][initial],y[train][initial])
    before=float((active.predict(x[test])==y[test]).mean())
    queried=nil.learning.query_uncertain(active,x[train],budget=12,excluded=initial.tolist())
    known=np.r_[initial,queried];active.fit(x[train][known],y[train][known])
    after=float((active.predict(x[test])==y[test]).mean())
    few=nil.learning.prototypes(x[train][initial],y[train][initial],x[test])
    reservoir=nil.learning.ReplayMemory(12,seed=7)
    for row in known:reservoir.add(dict(train_index=int(row),label=int(y[train][row])))
    loaded=nil.learning.ReplayMemory.load(reservoir.save(output/'replay'))
    assert loaded.records()==reservoir.records()
    rng=np.random.default_rng(7);features=rng.normal(size=(1200,3));p=1/(1+np.exp(-.4*features[:,0]))
    t=rng.binomial(1,p);outcome=3*t+2*features[:,0]+rng.normal(scale=.3,size=len(features))
    causal=nil.probabilistic.causal_ate(features,t,outcome)
    if abs(causal['ate']-3)>.12:raise AssertionError('Known-effect simulation reference failed')
    bayes=nil.probabilistic.BayesianLinear(noise_variance=.1).fit(features,outcome-3*t)
    np.testing.assert_array_equal(nil.probabilistic.BayesianLinear.load(bayes.save(output/'bayesian-linear')).predict(features[:5])['mean'],bayes.predict(features[:5])['mean'])
    report=dict(status='passed',data='sklearn bundled Iris + explicit synthetic causal simulation',
        held_out_iris_samples=len(test),semi_supervised_accuracy=semi_score,active_before=before,active_after=after,
        active_labels_added=len(queried),few_shot_accuracy=float((few['labels']==y[test]).mean()),
        causal_ate=causal['ate'],causal_interval=causal['interval'],causal_true_effect=3.,
        limitations='No causal identification from data alone; no guarantee of active/semi-supervised improvement.')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(exercise(args.output),indent=2))
