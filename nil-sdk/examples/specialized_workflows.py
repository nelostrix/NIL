"""Run NIL's scoped specialized workflows; --full requires explicit optional backends."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression,Ridge
import nil


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--full',action='store_true'); args=parser.parse_args()
    rng=np.random.default_rng(7)
    with tempfile.TemporaryDirectory(prefix='nil-specialized-') as directory:
        root=Path(directory)
        forecast=nil.forecasting.fit(np.arange(60.),lags=6,horizon=3,estimator=LinearRegression())
        np.testing.assert_allclose(forecast.predict(),[60,61,62],atol=1e-7)
        restored=nil.load(nil.save(forecast,root/'forecast'),trusted=True)
        np.testing.assert_array_equal(restored.predict(),forecast.predict())
        backtest=nil.forecasting.backtest(np.arange(80.),initial_train=40,horizon=3,lags=6,gap=2,step=10,estimator=LinearRegression())
        assert backtest['rmse']<1e-7
        print('PASS forecasting and rolling-origin gap:',backtest['rmse'])

        relevance=np.array([3,0,1,0,2,1]); scores=np.array([.9,.1,.2,.1,.7,.5]); groups=['a','b','a','b','a','b']
        print('PASS query metrics:',nil.ranking.evaluate(relevance,scores,groups,k=3))
        interactions=sparse.csr_matrix([[1,1,0],[1,0,1],[0,1,1]])
        recommender=nil.recommendation.fit(interactions,neighbors=2)
        assert recommender.recommend(0,k=1)[0][0]==2
        heldout=sparse.csr_matrix([[0,0,1],[0,1,0],[1,0,0]])
        assert nil.recommendation.evaluate(recommender,heldout,k=1)['recall@1']==1
        print('PASS sparse recommendations and cold start:',recommender.recommend(None,k=2))

        x=rng.normal(size=(100,3)); y=2*x[:,0]+3*x[:,1]
        regression=LinearRegression().fit(x[:50],y[:50])
        explanation=nil.inspection.shap_values(regression,x[50:52],x[:8],permutations=4)
        np.testing.assert_allclose(explanation.base_values+explanation.values.sum(axis=1),regression.predict(x[50:52]))
        print('PASS permutation Shapley additivity:',explanation.feature_names)
        noisy=y+rng.normal(size=len(y))
        intervals=nil.uncertainty.calibrate_weighted(regression,x[50:80],noisy[50:80],np.ones(30))
        assert intervals.interval(x[80:],np.ones(20)).shape==(20,2)
        multi=LinearRegression().fit(x[:50],np.column_stack([noisy[:50],2*noisy[:50]]))
        calibrated=nil.uncertainty.calibrate_multioutput(multi,x[50:80],np.column_stack([noisy[50:80],2*noisy[50:80]]))
        assert calibrated.interval(x[80:]).shape==(20,2,2)
        print('PASS weighted and multioutput uncertainty')

        schema=nil.datasets.Schema({'x':'int'})
        snapshot=nil.datasets.snapshot([{'x':1},{'x':2}],root/'dataset',schema=schema)
        evolved=nil.datasets.Schema({'x':'float','tag':{'type':'string','nullable':True}},version=2)
        assert nil.datasets.load_snapshot(snapshot,schema=evolved).records[0]=={'x':1.,'tag':None}
        with nil.datasets.StreamStore(root/'stream.sqlite') as store:
            receipt=store.append('source',0,[{'x':1}],schema=schema,transaction_id='batch-one')
            assert store.append('source',0,[{'x':1}],schema=schema,transaction_id='batch-one')==receipt
            assert store.offset('source')==1
        print('PASS typed snapshot migration and atomic source offset')

        with nil.selection.distributed_search(Ridge(),x,y,{'alpha':[.01,.1,1.]},storage=root/'search.sqlite',max_trials=3,cv=2,max_fits=6) as search:
            search.run(Ridge(),x,y,cv_plan=search.cv_plan,scoring=search.scoring,worker='example')
            assert all(trial['status']=='completed' for trial in search.results())
            print('PASS durable random search:',search.best()['params'])

        if args.full:
            for backend in ('lightgbm','catboost'):
                for task in ('classifier','regressor'):
                    parameters={'n_estimators':10,'min_child_samples':2} if backend=='lightgbm' else {'iterations':10,'depth':3}
                    model=nil.pipeline(nil.model(f'{backend}_{task}',**parameters),preprocess=nil.preprocessing.scaler())
                    target=(y>0).astype(int) if task=='classifier' else y
                    fitted=nil.fit(model,x[:80],target[:80],output_dir=root/'runs')
                    loaded=nil.load(nil.save(fitted,root/f'{backend}-{task}'),trusted=True)
                    np.testing.assert_allclose(loaded.predict(x[80:]),fitted.predict(x[80:]))
                    print('PASS native pipeline roundtrip:',backend,task)
            native=nil.runtime.require('lightgbm','lightgbm').LGBMRanker(n_estimators=5,n_jobs=1,min_child_samples=1,verbosity=-1)
            ranker=nil.ranking.fit(x[:6],relevance,groups,estimator=native)
            assert ranker.predict(x[:6]).shape==(6,)
            times=rng.exponential(np.exp(-x[:,0])); censor=rng.exponential(2,size=len(x))
            durations=np.minimum(times,censor)+.01; events=times<=censor
            survival=nil.survival.fit(x[:80],durations[:80],events[:80])
            print('PASS censored survival held-out metrics:',nil.survival.evaluate(survival,x[80:],durations[80:],events[80:]))
        print('All requested example paths passed; --full includes optional native integrations.')


if __name__=='__main__': main()
