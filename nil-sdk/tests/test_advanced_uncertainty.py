import numpy as np
import pytest
pytest.importorskip('sklearn')
from sklearn.dummy import DummyRegressor
import nil


def zero(outputs=1):
    y=np.zeros((2,outputs)) if outputs>1 else np.zeros(2)
    return DummyRegressor().fit(np.zeros((2,1)),y)


def test_multioutput_simultaneous_radius_and_persistence(tmp_path):
    rng=np.random.default_rng(7); x=np.zeros((199,1)); y=rng.normal(size=(199,2))*[1,3]
    calibrated=nil.uncertainty.calibrate_multioutput(zero(2),x,y,scales=[1,3])
    expected=nil.uncertainty._quantile(np.max(np.abs(y)/[1,3],axis=1),.1)
    assert calibrated.radius_==expected
    loaded=nil.load(nil.save(calibrated,tmp_path/'intervals'),trusted=True)
    np.testing.assert_array_equal(loaded.interval(x[:3]),calibrated.interval(x[:3]))
    test=rng.normal(size=(3000,2))*[1,3]; intervals=calibrated.interval(np.zeros((len(test),1)))
    coverage=((test>=intervals[:,:,0])&(test<=intervals[:,:,1])).all(axis=1).mean()
    assert coverage>.85


def test_mondrian_group_conditionals_and_unknown_groups():
    x=np.zeros((40,1)); y=np.r_[np.arange(20),np.arange(20)*10]
    calibrated=nil.uncertainty.calibrate_groups(zero(),x,y,['a']*20+['b']*20)
    interval=calibrated.interval(x[:2],['a','b'])
    assert interval[1,1]==10*interval[0,1]
    with pytest.raises(ValueError,match='Unknown'): calibrated.interval(x[:1],['c'])
    with pytest.raises(ValueError,match='too small'): nil.uncertainty.calibrate_groups(zero(),x[:2],y[:2],['a','b'])


def test_weighted_covariate_shift_quantiles_include_test_infinite_atom():
    x=np.zeros((20,1)); y=np.arange(20,dtype=float)
    calibrated=nil.uncertainty.calibrate_weighted(zero(),x,y,np.ones(20))
    ordinary=nil.uncertainty.calibrate(zero(),x,y)
    np.testing.assert_array_equal(calibrated.interval(x[:1],[1]),ordinary.interval(x[:1]))
    assert np.isinf(calibrated.interval(x[:1],[100])[0,1])
    with pytest.raises(ValueError): calibrated.interval(x[:1],[0])
    rng=np.random.default_rng(71); source=(rng.uniform(size=2000)<.2).astype(int)
    y=rng.normal(size=2000)*(1+source*4); weights=np.where(source==1,4,.25)
    shifted=nil.uncertainty.calibrate_weighted(zero(),source[:,None],y,weights)
    target=(rng.uniform(size=3000)<.8).astype(int); actual=rng.normal(size=3000)*(1+target*4)
    bounds=shifted.interval(target[:,None],np.where(target==1,4,.25))
    assert ((actual>=bounds[:,0])&(actual<=bounds[:,1])).mean()>.85


def test_block_exchangeable_scores_and_shape_contract():
    x=np.zeros((60,1)); y=np.repeat(np.arange(20,dtype=float),3)
    calibrated=nil.uncertainty.calibrate_blocks(zero(),x,y,block_size=3)
    assert calibrated.radius_==nil.uncertainty._quantile(np.arange(20),.1)
    assert calibrated.interval(x[:3]).shape==(3,2)
    with pytest.raises(ValueError,match='complete block'): calibrated.interval(x[:2])
    with pytest.raises(ValueError,match='complete nonoverlapping'): nil.uncertainty.calibrate_blocks(zero(),x[:-1],y[:-1],block_size=3)


@pytest.mark.parametrize('method',['groups','weighted','blocks'])
def test_calibrated_wrappers_roundtrip_with_their_required_context(tmp_path,method):
    x=np.zeros((40,1)); y=np.arange(40.)
    if method=='groups':
        model=nil.uncertainty.calibrate_groups(zero(),x,y,['a']*20+['b']*20)
        query=x[:2]; options={'groups':['a','b']}
    elif method=='weighted':
        model=nil.uncertainty.calibrate_weighted(zero(),x,y,np.ones(40))
        query=x[:2]; options={'test_likelihood_ratios':[1.,2.]}
    else:
        model=nil.uncertainty.calibrate_blocks(zero(),x,y,block_size=2)
        query=x[:2]; options={}
    loaded=nil.load(nil.save(model,tmp_path/method),trusted=True)
    np.testing.assert_array_equal(loaded.interval(query,**options),model.interval(query,**options))
    assert nil.uncertainty.evaluate(loaded,query,y[:2],**options)['count']==2
