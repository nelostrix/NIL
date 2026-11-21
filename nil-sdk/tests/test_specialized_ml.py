import copy
import numpy as np
import pytest

pytest.importorskip('sklearn')
import nil
from sklearn.linear_model import LinearRegression,Ridge
from scipy import sparse


@pytest.mark.parametrize('backend', ['lightgbm','catboost'])
@pytest.mark.parametrize('task', ['classifier','regressor'])
def test_optional_boosting_native_pipeline_roundtrip(tmp_path, backend, task):
    pytest.importorskip(backend)
    rng=np.random.default_rng(7); x=rng.normal(size=(80,3)); y=(x[:,0]>0).astype(int) if task=='classifier' else x[:,0]*2
    options={'n_estimators':10, 'min_child_samples':2} if backend=='lightgbm' else {'iterations':10, 'depth':3}
    model=nil.pipeline(nil.model(f'{backend}_{task}',**options),preprocess=nil.preprocessing.scaler())
    fitted=nil.fit(model,x[:60],y[:60],output_dir=tmp_path/'runs')
    predictions=fitted.predict(x[60:])
    loaded=nil.load(nil.save(fitted,tmp_path/f'{backend}-{task}'),trusted=True)
    np.testing.assert_allclose(loaded.predict(x[60:]),predictions)


def test_direct_forecasting_and_gap_backtests_do_not_leak(tmp_path):
    from datetime import date,timedelta
    values=np.arange(60,dtype=float)
    dates=[(date(2020,1,1)+timedelta(days=i)).isoformat() for i in range(60)]
    model=nil.model('lag_forecaster',lags=4,horizon=3,estimator=LinearRegression()).fit(values[:40])
    np.testing.assert_allclose(model.predict(),[40,41,42],atol=1e-9)
    clone= nil.load(nil.save(model,tmp_path/'forecast'),trusted=True)
    np.testing.assert_allclose(clone.predict(),model.predict())
    report=nil.forecasting.backtest(values,lags=4,horizon=3,gap=2,initial_train=30,step=7,estimator=LinearRegression(),dates=dates)
    assert report['rmse']<1e-7
    assert all(row['train_end']+2==row['test_start'] for row in report['folds'])
    altered=values.copy(); altered[32:]+=10000
    first=nil.forecasting.backtest(altered,lags=4,horizon=3,gap=2,initial_train=30,step=50,estimator=LinearRegression())['folds'][0]
    np.testing.assert_allclose(first['prediction'],[32,33,34],atol=1e-7)
    with pytest.raises(ValueError): nil.forecasting.fit(values,dates=dates[::-1])
    with pytest.raises(ValueError): model.predict(4)


def test_query_metrics_and_actual_ranker_roundtrip(tmp_path):
    from sklearn.metrics import ndcg_score
    y=np.array([3,0,1,0,2,1]); score=np.array([.9,.1,.2,.1,.7,.5]); groups=np.array(['a','b','a','b','a','b'])
    metrics=nil.ranking.evaluate(y,score,groups,k=3)
    expected=np.mean([ndcg_score([2**y[groups==g]-1],[score[groups==g]],k=3) for g in ['a','b']])
    assert metrics['ndcg@3']==pytest.approx(expected)
    assert nil.ranking.evaluate([0,0],[1,0],['a','a'])['map@10']==0
    pytest.importorskip('lightgbm')
    x=np.column_stack([y,np.arange(6)])
    ranker=nil.model('query_ranker',estimator=nil.runtime.require('lightgbm','lightgbm').LGBMRanker(n_estimators=5,min_child_samples=1,n_jobs=1,verbosity=-1)).fit(x,y,groups=groups)
    loaded=nil.load(nil.save(ranker,tmp_path/'ranker'),trusted=True)
    np.testing.assert_allclose(loaded.predict(x),ranker.predict(x))
    with pytest.raises(ValueError): nil.ranking.evaluate(y,score[:-1],groups)


def test_sparse_recommendation_cold_start_budgets_and_persistence(tmp_path):
    matrix=sparse.csr_matrix([[1,1,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]])
    model=nil.model('item_cosine',neighbors=2).fit(matrix)
    assert model.recommend(0,k=1)[0][0]==2
    assert model.recommend(None,k=2)==[(0,2.),(1,2.)]
    assert model.recommend(3,k=2)==model.recommend(None,k=2)
    loaded=nil.load(nil.save(model,tmp_path/'recommend'),trusted=True)
    assert loaded.recommend(0,k=2)==model.recommend(0,k=2)
    with pytest.raises(ValueError,match='max_pairs'): nil.recommendation.fit(sparse.csr_matrix(np.ones((3,100))),max_pairs=10)
    with pytest.raises(TypeError): nil.recommendation.fit(np.ones((3,3)))


def test_survival_native_parity_censoring_and_roundtrip(tmp_path):
    lifelines=pytest.importorskip('lifelines'); pd=pytest.importorskip('pandas')
    rng=np.random.default_rng(42); x=rng.normal(size=(150,2)); times=rng.exponential(np.exp(-x[:,0])); censor=rng.exponential(2,size=150)
    durations=np.minimum(times,censor)+.01; events=times<=censor
    model=nil.model('cox_ph').fit(x[:100],(durations[:100],events[:100]))
    frame=pd.DataFrame(x[:100],columns=['x0','x1']); frame['__time']=durations[:100]; frame['__event']=events[:100]
    reference=lifelines.CoxPHFitter(penalizer=.1).fit(frame,'__time','__event')
    np.testing.assert_allclose(model.predict(x[100:]),reference.predict_partial_hazard(pd.DataFrame(x[100:],columns=['x0','x1'])))
    assert nil.survival.evaluate(model,x[100:],durations[100:],events[100:])['concordance_index']>.6
    sf=model.survival_function(x[100:103],[0,.5,1,2]); assert np.all(np.diff(sf,axis=1)<=0)
    loaded=nil.load(nil.save(model,tmp_path/'cox'),trusted=True); np.testing.assert_allclose(loaded.predict(x),model.predict(x))
    km=nil.model('kaplan_meier').fit([1,2,3,4],[1,0,1,0]); np.testing.assert_allclose(km.predict([0,1,2,3]),[1,.75,.75,.375])
    with pytest.raises(ValueError): nil.survival.fit(x[:4],[1,-2,3,4],[1,1,0,0])


def test_bounded_permutation_shap_linear_exact_names_and_sparse_budget():
    pd=pytest.importorskip('pandas')
    x=pd.DataFrame({'first':[0.,1.,2.,3.],'second':[1.,0.,3.,2.]}); y=2*x['first']+3*x['second']
    model=nil.pipeline(LinearRegression(),preprocess=nil.preprocessing.scaler()); model.fit(x,y)
    result=nil.inspection.shap_values(model,x.iloc[:2],x,permutations=4)
    assert result.feature_names==['first','second']
    np.testing.assert_allclose(result.base_values+result.values.sum(axis=1),model.predict(x.iloc[:2]))
    np.testing.assert_allclose(result.values, (x.iloc[:2]-x.mean()).to_numpy()*[2,3],atol=1e-8)
    with pytest.raises(ValueError,match='max_evaluations'): nil.inspection.shap_values(model,x,x,max_evaluations=2)
    sparse_model=LinearRegression().fit(sparse.eye(100),np.arange(100.))
    with pytest.raises(ValueError,match='budget'): nil.inspection.shap_values(sparse_model,sparse.eye(100).tocsr()[:1],sparse.eye(100).tocsr()[:2],max_features=200,max_dense_bytes=20)


def test_recommendation_heldout_metrics_refuse_seen_item_leakage():
    matrix=sparse.csr_matrix([[1,1,0],[1,0,1],[0,1,1]])
    model=nil.recommendation.fit(matrix)
    heldout=sparse.csr_matrix([[0,0,1],[0,1,0],[1,0,0]])
    assert nil.recommendation.evaluate(model,heldout,k=1)['recall@1']==1
    with pytest.raises(ValueError,match='disjoint'): nil.recommendation.evaluate(model,matrix,k=1)


def test_query_split_keeps_entire_queries_on_one_side():
    groups=np.repeat(np.arange(10),3); x=np.arange(60).reshape(30,2); relevance=np.tile([0,1,2],10)
    split=nil.ranking.split(x,relevance,groups,test_size=.3)
    assert set(groups[split.train_indices]).isdisjoint(groups[split.test_indices])


def test_failed_forecast_refit_preserves_last_valid_model():
    model=nil.forecasting.fit(np.arange(30.),lags=3,horizon=2)
    before=model.predict().copy()
    with pytest.raises(ValueError): model.fit(np.arange(30.),dates=['bad-date']*30)
    np.testing.assert_array_equal(model.predict(),before)


def test_nonlinear_shap_additivity_and_probability_output():
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LogisticRegression
    rng=np.random.default_rng(7); x=rng.normal(size=(50,3)); y=x[:,0]*x[:,1]+x[:,2]
    tree=DecisionTreeRegressor(max_depth=3,random_state=0).fit(x,y)
    explanation=nil.inspection.shap_values(tree,x[:3],x[3:8],permutations=20)
    np.testing.assert_allclose(explanation.values.sum(axis=1)+explanation.base_values,tree.predict(x[:3]))
    classifier=LogisticRegression().fit(x,(y>0).astype(int))
    result=nil.inspection.shap_values(classifier,x[:2],x[3:8],class_index=1)
    np.testing.assert_allclose(result.base_values+result.values.sum(axis=1),classifier.predict_proba(x[:2])[:,1])


def test_shap_transformed_one_hot_names_and_actual_sparse_pipeline():
    pd=pytest.importorskip('pandas')
    frame=pd.DataFrame({'age':[20.,30.,40.,50.,60.,70.],'city':['a','b','a','b','a','b']})
    preprocessing=nil.preprocessing.tabular(numeric_columns=['age'],categorical_columns=['city'],sparse=True)
    model=nil.pipeline(Ridge(solver='lsqr'),preprocess=preprocessing).fit(frame,np.array([1.,3.,4.,2.,5.,6.]))
    explanation=nil.inspection.shap_values(model,frame.iloc[:2],frame.iloc[2:],permutations=4)
    assert any('city_a' in name for name in explanation.feature_names)
    assert explanation.values.shape[1]==len(explanation.feature_names)
    np.testing.assert_allclose(explanation.base_values+explanation.values.sum(axis=1),model.predict(frame.iloc[:2]))


def test_shap_refuses_reordered_dataframe_background():
    pd=pytest.importorskip('pandas')
    frame=pd.DataFrame({'first':[1.,2.,3.],'second':[3.,1.,2.]})
    model=LinearRegression().fit(frame,[1.,2.,3.])
    with pytest.raises(ValueError,match='columns'):
        nil.inspection.shap_values(model,frame,frame[['second','first']])


def test_kaplan_meier_bundle_keeps_censoring_curve(tmp_path):
    pytest.importorskip('lifelines')
    model=nil.survival.kaplan_meier([1,2,3,4],[1,0,1,0])
    loaded=nil.load(nil.save(model,tmp_path/'km'),trusted=True)
    np.testing.assert_array_equal(loaded.predict([0,1,2,3]),model.predict([0,1,2,3]))
