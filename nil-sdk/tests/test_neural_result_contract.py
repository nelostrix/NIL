"""Actual train/predict/evaluate boundary, including restoration after failure."""
import json
import pytest
torch=pytest.importorskip('torch')
import nil


def test_training_result_shared_entrypoints(tmp_path):
    torch.set_num_threads(1)
    x=torch.tensor([[0.],[1.],[2.],[3.]])
    data=nil.data.tensors(x,2*x)
    result=nil.train(torch.nn.Linear(1,1),data,epochs=2,batch_size=2,device='cpu',output_dir=tmp_path)
    result.model.train()
    torch.testing.assert_close(nil.predict(result,x,device_name='cpu'),nil.predict(result.model,x,device_name='cpu'))
    assert nil.evaluate(result,data,device_name='cpu')==nil.evaluate(result.model,data,device_name='cpu')
    assert result.model.training
    info=nil.contracts.result_info(result)
    assert info.family=='neural' and info.status=='completed' and info.history_length==2
    assert isinstance(info.checkpoint,str)
    assert 'model' not in info.to_dict()
    json.dumps(info.to_dict())
    with pytest.raises(RuntimeError):nil.predict(result,torch.zeros(2,3),device_name='cpu')
    assert result.model.training
    result.model.eval()
    nil.predict(result,x,device_name='cpu')
    assert not result.model.training
