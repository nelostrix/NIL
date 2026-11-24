"""Local native model protocols, no model or network downloads."""
import json
import numpy as np
import pytest
import torch
hf=pytest.importorskip('transformers')
from tokenizers import Tokenizer,models,pre_tokenizers
import nil


def tokenizer(root):
    backend=Tokenizer(models.WordLevel({'[UNK]':0,'[PAD]':1,'cat':2,'dog':3,'red':4},unk_token='[UNK]'))
    backend.pre_tokenizer=pre_tokenizers.Whitespace()
    result=hf.PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',pad_token='[PAD]')
    result.save_pretrained(root);return result


def test_local_text_encoder_roundtrip_integrity_and_missing_path(tmp_path):
    root=tmp_path/'source';root.mkdir();tokenizer(root)
    torch.manual_seed(1)
    hf.BertModel(hf.BertConfig(vocab_size=5,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24)).save_pretrained(root)
    encoder=nil.rag.load_encoder(root);encoder.model.train()
    expected=encoder.encode(['cat','dog'])
    assert expected.shape==(2,16) and encoder.model.training
    destination=encoder.save(tmp_path/'saved')
    restored=nil.rag.load_encoder(destination)
    np.testing.assert_array_equal(expected,restored.encode(['cat','dog']))
    with pytest.raises(FileExistsError):encoder.save(destination)
    with (destination/'config.json').open('a') as stream:stream.write(' ')
    with pytest.raises(ValueError,match='integrity'):nil.rag.load_encoder(destination)
    with pytest.raises(ValueError):nil.rag.load_encoder(tmp_path/'missing')


def test_ctc_real_native_forward_local_reload_and_contract(tmp_path):
    root=tmp_path/'ctc';root.mkdir()
    (root/'vocab.json').write_text(json.dumps({'[PAD]':0,'[UNK]':1,'|':2,'A':3,'B':4}))
    tok=hf.Wav2Vec2CTCTokenizer(str(root/'vocab.json'),unk_token='[UNK]',pad_token='[PAD]',word_delimiter_token='|')
    processor=hf.Wav2Vec2Processor(hf.Wav2Vec2FeatureExtractor(sampling_rate=16000,return_attention_mask=True),tok)
    processor.save_pretrained(root)
    config=hf.Wav2Vec2Config(vocab_size=5,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,
        conv_dim=(8,8,8),conv_stride=(5,2,2),conv_kernel=(10,3,3),num_conv_pos_embeddings=8,num_conv_pos_embedding_groups=2,mask_time_prob=0.)
    hf.Wav2Vec2ForCTC(config).save_pretrained(root)
    recognizer=nil.audio.load_asr(root);recognizer.model.train()
    waveform=nil.audio.Waveform(np.zeros((1,4000),dtype=np.float32),16000)
    expected=recognizer.transcribe(waveform)
    assert isinstance(expected['text'],str) and recognizer.model.training
    restored=nil.audio.load_asr(recognizer.save(tmp_path/'ctc-saved'))
    assert restored.transcribe(waveform)['text']==expected['text']
    with pytest.raises(ValueError):restored.transcribe(nil.audio.Waveform(np.zeros((2,4000),dtype=np.float32),16000))


def test_clip_local_native_joint_embeddings_and_reload(tmp_path):
    from PIL import Image
    root=tmp_path/'clip';root.mkdir();tok=tokenizer(root)
    hf.CLIPProcessor(hf.CLIPImageProcessor(size={'shortest_edge':16},crop_size={'height':16,'width':16}),tok).save_pretrained(root)
    config=hf.CLIPConfig(text_config=dict(vocab_size=5,hidden_size=16,intermediate_size=24,num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=16,pad_token_id=1,bos_token_id=0,eos_token_id=0),
        vision_config=dict(hidden_size=16,intermediate_size=24,num_hidden_layers=1,num_attention_heads=2,image_size=16,patch_size=8),projection_dim=8)
    hf.CLIPModel(config).save_pretrained(root)
    encoder=nil.multimodal.load_encoder(root)
    images=[Image.new('RGB',(16,16),'red')];text=encoder.encode_text(['red'])
    visual=encoder.encode_images(images)
    assert text.shape==visual.shape==(1,8)
    np.testing.assert_allclose(np.linalg.norm(visual,axis=1),1,atol=1e-6)
    restored=nil.multimodal.load_encoder(encoder.save(tmp_path/'clip-saved'))
    np.testing.assert_array_equal(text,restored.encode_text(['red']))
    np.testing.assert_array_equal(visual,restored.encode_images(images))
