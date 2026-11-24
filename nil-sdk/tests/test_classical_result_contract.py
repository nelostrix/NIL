"""Real classical fit result metadata and shared dispatch."""
import json
import pytest
pytest.importorskip('sklearn')
import nil


def test_classical_result_contract(tmp_path):
    x=[[0.],[1.],[2.],[3.]];y=[1.,3.,5.,7.]
    result=nil.fit(nil.model('linear_regression'),x,y,output_dir=tmp_path)
    info=nil.contracts.result_info(result)
    assert info.family=='classical' and info.model_count==1 and info.status=='completed'
    assert info.metric_names==tuple(sorted(result.metrics))
    assert info.checkpoint is None and info.run_path is not None
    json.dumps(info.to_dict())
    assert nil.predict(result,x).tolist()==nil.predict(result.model,x).tolist()
    assert nil.evaluate(result,x,y)==nil.evaluate(result.model,x,y)
