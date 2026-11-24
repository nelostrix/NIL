"""CPU audio-event and audio/video fusion classification, manifests and reload."""
import argparse
import json
from pathlib import Path
import wave


def exercise(output):
    import nil
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(1);torch.manual_seed(7)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sound=[];combined=[];labels=[]
    for i in range(20):
        label=i%2;frequency=(300 if label==0 else 1600)+i
        samples=.5*np.sin(2*np.pi*frequency*np.arange(1600)/8000)
        wav=output/f'{i}.wav'
        with wave.open(str(wav),'wb') as stream:
            stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(8000)
            stream.writeframes(np.round(samples*32767).astype('<i2').tobytes())
        path=output/f'{i}.gif';color=(255,0,0) if label==0 else (0,0,255)
        Image.new('RGB',(12,8),color).save(path,save_all=True,append_images=[Image.new('RGB',(12,8),(0,0,0))],duration=[100,100],loop=0)
        audio=nil.audio.load_wav(wav);video=nil.multimodal.load_clip(path,size=(8,8))
        nil.multimodal.aligned_audio(video,audio)
        a=nil.audio.features(audio);v=nil.multimodal.visual_features(video)
        sound.append(a);combined.append(nil.multimodal.fuse_features(dict(audio=a,video=v),dict(audio=66,video=9)));labels.append(label)
    y=torch.tensor(labels);reports={}
    for name,values in [('audio',sound),('fusion',combined)]:
        x=torch.tensor(np.stack(values));model=nil.nn.mlp(x.shape[1],2,hidden=(8,))
        result=nil.train(model,nil.data.tensors(x[:12],y[:12]),task='classification',epochs=35,
            batch_size=12,learning_rate=.03,device='cpu',output_dir=output/name)
        metrics=nil.evaluate(result,nil.data.tensors(x[12:],y[12:]),task='classification',device_name='cpu')
        if metrics['accuracy']<.9:raise AssertionError(f'{name}: declared held-out quality floor failed')
        restored=nil.nn.mlp(x.shape[1],2,hidden=(8,));nil.training.checkpoint.load(result.checkpoint,restored)
        torch.testing.assert_close(nil.predict(restored,x[12:],device_name='cpu'),nil.predict(result,x[12:],device_name='cpu'),rtol=0,atol=0)
        reports[name]=dict(metrics,exact_reload=True)
    manifest=dict(schema_version=1,classes=['low_tone_red','high_tone_blue'],audio=dict(sample_rate=8000,bands=32,n_fft=512,hop=256),
                  video=dict(codec='GIF',size=[8,8],max_frames=64),fusion_dimensions=dict(audio=66,video=9),missing='error')
    nil.data.snapshot([manifest],output/'processor-manifest',max_records=1)
    assert nil.data.load_snapshot(output/'processor-manifest')[0]==manifest
    report=dict(status='passed',scope='Synthetic tone/color classification wiring and codecs; not speech recognition, video understanding or natural-data quality',
                training_samples=12,held_out_samples=8,results=reports,manifest=manifest)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(exercise(args.output),indent=2))
