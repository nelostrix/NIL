"""Explicit offline CPU acceptance of user-prepared native model directories.

No downloads or package installation. Synthetic vision/text probes and unlabeled
ASR audio verify execution, not broad pretrained quality or speech intelligibility.
"""
import argparse
import gc
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import nil,numpy as np,torch
    from PIL import Image
    from importlib.metadata import version
    report=dict(schema_version=1,scope='Offline CPU execution/reload; limited probes, not general quality',
                versions={name:version(name) for name in ('torch','transformers','numpy','Pillow')},results={})
    def record(name,start,**values):report['results'][name]=dict(seconds=time.monotonic()-start,**values)
    start=time.monotonic();encoder=nil.rag.load_encoder(args.models/'text')
    index=nil.rag.semantic_index(encoder,mode='dense',min_similarity=.1)
    index.add_many([dict(id='pet',text='A kitten is resting on a sofa.'),dict(id='space',text='The spacecraft orbits the distant planet.')])
    hits=index.search('a cat sleeping indoors',k=1);assert hits[0]['document_id']=='pet'
    expected=encoder.encode(['cat','planet']);saved=encoder.save(args.output/'text')
    del encoder,index;gc.collect();encoder=nil.rag.load_encoder(saved)
    np.testing.assert_array_equal(expected,encoder.encode(['cat','planet']))
    record('text',start,hit=hits[0]['document_id'],cosine=hits[0]['dense_score'],reload_exact=True)
    del encoder;gc.collect()
    start=time.monotonic();encoder=nil.multimodal.load_encoder(args.models/'vision')
    image=[Image.new('RGB',(64,64),'red')];texts=['a red image','a blue image']
    visual=encoder.encode_images(image);text=encoder.encode_text(texts)
    saved=encoder.save(args.output/'vision');del encoder;gc.collect();encoder=nil.multimodal.load_encoder(saved)
    visual_reload=encoder.encode_images(image);text_reload=encoder.encode_text(texts)
    np.testing.assert_allclose(visual,visual_reload,atol=1e-6,rtol=1e-6);np.testing.assert_allclose(text,text_reload,atol=1e-6,rtol=1e-6)
    record('vision',start,shape=list(visual.shape),synthetic_similarity=(visual@text.T).tolist(),reload_atol=1e-6,reload_max_absolute_error=float(max(np.abs(visual-visual_reload).max(),np.abs(text-text_reload).max())))
    del encoder;gc.collect()
    start=time.monotonic();model=nil.audio.load_tts(args.models/'tts')
    state=torch.random.get_rng_state().clone();text='NIL makes local model workflows explicit.'
    wave=model.synthesize(text,seed=42);repeat=model.synthesize(text,seed=42)
    assert np.array_equal(wave.samples,repeat.samples) and torch.equal(state,torch.random.get_rng_state())
    nil.audio.save_wav(wave,args.output/'speech.wav');saved=model.save(args.output/'tts')
    del model;gc.collect();model=nil.audio.load_tts(saved)
    np.testing.assert_array_equal(wave.samples,model.synthesize(text,seed=42).samples)
    record('tts',start,duration=wave.seconds,reload_exact=True,caller_rng_preserved=True,
           fixture_license='CC-BY-NC-4.0; research/noncommercial weights, not distributed with NIL',intelligibility_evaluated=False)
    del model;gc.collect()
    start=time.monotonic();model=nil.audio.load_asr(args.models/'asr')
    waveform=nil.audio.load_wav(args.audio)
    # Explicit headroom for the polyphase interpolation filter; no hidden clipping.
    waveform=nil.audio.Waveform(waveform.samples*.8,waveform.sample_rate)
    waveform=nil.audio.resample(nil.audio.mono(waveform),model.sample_rate,method='polyphase')
    text=model.transcribe(waveform)['text'];assert text.strip()
    saved=model.save(args.output/'asr');del model;gc.collect();model=nil.audio.load_asr(saved)
    assert model.transcribe(waveform)['text']==text
    record('asr',start,text=text,duration=waveform.seconds,reload_exact=True,wer=None,reference_transcript_available=False)
    # Pin preparation identities/license rather than copying any model into the SDK.
    report['model_manifests']={name:json.loads((args.models/name/'.download-manifest.json').read_text()) for name in ('text','vision','tts','asr')}
    (args.output/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['results'],indent=2))


if __name__=='__main__':main()
