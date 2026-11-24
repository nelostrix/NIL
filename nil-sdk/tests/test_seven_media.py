"""Real codecs and signals: PCM, animation timing, alignment and learned fusion."""
from pathlib import Path
import wave
import pytest
np=pytest.importorskip('numpy');PIL=pytest.importorskip('PIL.Image');torch=pytest.importorskip('torch')
import nil


def wav(path, frequency=440, rate=8000, seconds=.2, channels=1, width=2):
    x=.5*np.sin(2*np.pi*frequency*np.arange(round(rate*seconds))/rate)
    values=np.tile(x[:,None],(1,channels))
    if width==1:raw=np.round(values*127+128).astype(np.uint8).tobytes()
    elif width==2:raw=np.round(values*32767).astype('<i2').tobytes()
    elif width==3:
        integers=np.round(values.reshape(-1)*((1<<23)-1)).astype(np.int32)
        raw=np.stack([integers&255,(integers>>8)&255,(integers>>16)&255],1).astype(np.uint8).tobytes()
    else:raw=np.round(values*((1<<31)-1)).astype('<i4').tobytes()
    with wave.open(str(path),'wb') as stream:
        stream.setnchannels(channels);stream.setsampwidth(width);stream.setframerate(rate);stream.writeframes(raw)
    return path


def clip(path,color=(255,0,0)):
    PIL.new('RGB',(12,8),color).save(path,save_all=True,append_images=[PIL.new('RGB',(12,8),(0,0,0))],duration=[100,100],loop=0)
    return path


@pytest.mark.parametrize('width',[1,2,3,4])
def test_pcm_channel_rate_roundtrip_and_features(tmp_path,width):
    audio=nil.audio.load_wav(wav(tmp_path/'tone.wav',channels=2,width=width))
    assert audio.samples.shape==(2,1600) and audio.sample_rate==8000 and audio.seconds==.2
    assert np.max(np.abs(audio.samples))==pytest.approx(.5,abs=.01)
    values=nil.audio.features(audio)
    assert values.shape==(66,) and np.isfinite(values).all()
    resampled=nil.audio.resample(audio,16000)
    assert resampled.samples.shape==(2,3200) and resampled.seconds==audio.seconds
    assert nil.audio.segment(audio,0,.1).samples.shape==(2,800)
    with pytest.raises(ValueError):nil.audio.segment(audio,0,.3)
    with pytest.raises(ValueError):nil.audio.load_wav(tmp_path/'tone.wav',max_seconds=.01)
    with pytest.raises(ValueError):nil.audio.load_wav(tmp_path/'tone.wav',max_bytes=10)


def test_animation_timing_budgets_and_alignment(tmp_path):
    video=nil.multimodal.load_clip(clip(tmp_path/'clip.gif'),size=(4,6))
    assert video.frames.shape==(2,3,4,6) and video.timestamps==(0.,.1) and video.durations==(.1,.1)
    audio=nil.audio.load_wav(wav(tmp_path/'sound.wav'))
    assert [part.samples.shape[1] for part in nil.multimodal.aligned_audio(video,audio)]==[800,800]
    with pytest.raises(ValueError,match='cover'):nil.multimodal.aligned_audio(video,nil.audio.segment(audio,0,.1))
    with pytest.raises(ValueError,match='frame budget'):nil.multimodal.load_clip(tmp_path/'clip.gif',max_frames=1)
    with pytest.raises(ValueError,match='memory budget'):nil.multimodal.load_clip(tmp_path/'clip.gif',max_decoded_bytes=10)
    assert nil.multimodal.visual_features(video).shape==(9,)


def test_fusion_missing_features_are_explicit():
    result=nil.multimodal.fuse_features({'image':[1,2]},dict(image=2,text=3),missing='zeros')
    np.testing.assert_array_equal(result,[1,2,0,0,0,1,0])
    with pytest.raises(ValueError,match='Missing'):nil.multimodal.fuse_features({'image':[1,2]},dict(image=2,text=3))
    with pytest.raises(ValueError):nil.multimodal.fuse_features({'image':[float('nan'),2]},dict(image=2))


def test_audio_and_video_fusion_classifier_actual_learning(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(7)
    features=[];labels=[]
    for i in range(16):
        label=i%2
        audio=nil.audio.load_wav(wav(tmp_path/f'{i}.wav',frequency=(300 if label==0 else 1600)+i))
        video=nil.multimodal.load_clip(clip(tmp_path/f'{i}.gif',color=(255,0,0) if label==0 else (0,0,255)),size=(8,8))
        features.append(nil.multimodal.fuse_features(dict(audio=nil.audio.features(audio),video=nil.multimodal.visual_features(video)),dict(audio=66,video=9)))
        labels.append(label)
    x=torch.tensor(np.stack(features));y=torch.tensor(labels)
    train=nil.data.tensors(x[:10],y[:10]);valid=nil.data.tensors(x[10:],y[10:])
    result=nil.train(nil.nn.mlp(x.shape[1],2,hidden=(8,)),train,task='classification',epochs=25,batch_size=10,learning_rate=.03,device='cpu',output_dir=tmp_path/'run')
    assert nil.evaluate(result,valid,task='classification',device_name='cpu')['accuracy']==1
    restored=nil.nn.mlp(x.shape[1],2,hidden=(8,));nil.training.checkpoint.load(result.checkpoint,restored)
    torch.testing.assert_close(nil.predict(result,x[10:],device_name='cpu'),nil.predict(restored,x[10:],device_name='cpu'),rtol=0,atol=0)
