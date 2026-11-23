"""Pinned model/tokenizer resolution and opt-in downloads without network access."""
from types import SimpleNamespace
import pytest
import nil.generative as generative


@pytest.mark.parametrize('download',[False,True])
def test_model_tokenizer_share_revision_cache_auth_and_network_contract(monkeypatch,tmp_path,download):
    calls=[]
    def model(path,**options):calls.append(('model',path,options));return object()
    def tokenizer(path,**options):
        calls.append(('tokenizer',path,options))
        return SimpleNamespace(pad_token_id=0,eos_token_id=0)
    backend=SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=model),AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer))
    monkeypatch.setattr(generative,'require',lambda *args:backend)
    generative.load_causal_lm('explicit/repo',allow_download=download,revision='f'*40,cache_dir=tmp_path,token='fixture-auth',use_safetensors=True)
    for kind,path,options in calls:
        assert path=='explicit/repo'
        assert options['revision']=='f'*40 and options['cache_dir']==str(tmp_path)
        assert options['local_files_only'] is (not download) and options['trust_remote_code'] is False
        assert options['token']=='fixture-auth'
    assert calls[0][2]['use_safetensors'] is True and 'use_safetensors' not in calls[1][2]


@pytest.mark.parametrize('options',[{'allow_download':'false'},{'trust_remote_code':True},{'local_files_only':False},{'revision':''}])
def test_ambiguous_or_remote_code_loading_is_refused(monkeypatch,options):
    backend=SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a,**k:pytest.fail('Unexpected model load')))
    monkeypatch.setattr(generative,'require',lambda *args:backend)
    with pytest.raises((ValueError,TypeError)):generative.load_causal_lm('explicit/repo',**options)
