"""Bounded local PCM WAV decoding and deterministic audio classification features."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import math
import wave
from nil.runtime import require


@dataclass(frozen=True)
class Waveform:
    samples: object
    sample_rate: int

    @property
    def seconds(self):return self.samples.shape[1]/self.sample_rate


def _waveform(value):
    np=require('numpy','audio')
    if not isinstance(value,Waveform) or type(value.sample_rate) is not int or not 1<=value.sample_rate<=192000:raise ValueError('Expected a Waveform with a valid rate')
    samples=np.asarray(value.samples,dtype=np.float32)
    if samples.ndim!=2 or not 1<=samples.shape[0]<=8 or not 1<=samples.shape[1]<=192000*60 or not np.isfinite(samples).all() or (np.abs(samples)>1).any():raise ValueError('Expected finite [channels,samples] amplitudes in [-1,1]')
    return samples


def load_wav(path, *, max_bytes=32*1024*1024, max_seconds=60.):
    np=require('numpy','audio')
    if type(max_bytes) is not int or not 1<=max_bytes<=256*1024*1024 or type(max_seconds) not in (int,float) or not math.isfinite(max_seconds) or not 0<max_seconds<=60:raise ValueError('Invalid WAV budgets')
    with Path(path).open('rb') as stream:encoded=stream.read(max_bytes+1)
    if len(encoded)>max_bytes:raise ValueError('WAV exceeds max_bytes')
    with wave.open(BytesIO(encoded),'rb') as source:
        channels=source.getnchannels();rate=source.getframerate();width=source.getsampwidth();frames=source.getnframes()
        if source.getcomptype()!='NONE' or width not in (1,2,3,4) or not 1<=channels<=8 or not 1<=rate<=192000 or not 1<=frames<=rate*max_seconds:raise ValueError('Unsupported PCM WAV or duration/channel budget')
        raw=source.readframes(frames)
    if len(raw)!=frames*channels*width:raise ValueError('Truncated WAV samples')
    if width==1:values=(np.frombuffer(raw,dtype=np.uint8).astype(np.float32)-128)/128
    elif width==3:
        b=np.frombuffer(raw,dtype=np.uint8).reshape(-1,3).astype(np.int32)
        signed=b[:,0]|(b[:,1]<<8)|(b[:,2]<<16);signed=(signed^(1<<23))-(1<<23)
        values=signed.astype(np.float32)/(1<<23)
    else:values=np.frombuffer(raw,dtype='<i2' if width==2 else '<i4').astype(np.float32)/(1<<(8*width-1))
    return Waveform(values.reshape(-1,channels).T.copy(),rate)


def resample(waveform, sample_rate, *, method="linear"):
    np=require('numpy','audio');x=_waveform(waveform)
    if type(sample_rate) is not int or not 1<=sample_rate<=192000:raise ValueError('Invalid target rate')
    count=max(1,round(x.shape[1]*sample_rate/waveform.sample_rate))
    if count>192000*60:raise ValueError('Resampled waveform exceeds budget')
    if method not in {'linear','polyphase'}:raise ValueError('Choose linear or polyphase resampling')
    if method=='polyphase':
        from math import gcd
        factor=gcd(waveform.sample_rate,sample_rate);up=sample_rate//factor;down=waveform.sample_rate//factor
        if max(up,down)>2048:raise ValueError('Polyphase ratio exceeds filter budget')
        signal=require('scipy.signal','signal')
        result=signal.resample_poly(x,up,down,axis=1)[:,:count]
        if result.shape[1]<count:result=np.pad(result,((0,0),(0,count-result.shape[1])))
        if np.max(np.abs(result))>1:raise ValueError('Resampling overshoot exceeds normalized amplitude; normalize input explicitly')
        return Waveform(result.astype(np.float32),sample_rate)
    old=np.arange(x.shape[1]);new=np.arange(count)*waveform.sample_rate/sample_rate
    # Linear interpolation is explicit and deterministic, not a bandlimited codec.
    return Waveform(np.stack([np.interp(new,old,row) for row in x]).astype(np.float32),sample_rate)


def segment(waveform, start, end):
    x=_waveform(waveform)
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (start,end)) or not 0<=start<end<=waveform.seconds:raise ValueError('Segment must lie inside the waveform')
    first=round(start*waveform.sample_rate);last=round(end*waveform.sample_rate)
    if first==last:raise ValueError('Segment contains no samples')
    return Waveform(x[:,first:last].copy(),waveform.sample_rate)


def features(waveform, *, bands=32, n_fft=512, hop=256):
    """Mean/std log-power bands plus RMS/ZCR; channel mean, no learned processor.

    Rates must match between training/inference; resample explicitly if needed.
    Features summarize temporal order and are not a speech recognition frontend.
    """
    np=require('numpy','audio');x=_waveform(waveform).mean(axis=0)
    if type(n_fft) is not int or not 16<=n_fft<=8192 or type(hop) is not int or not 1<=hop<=n_fft or type(bands) is not int or not 1<=bands<=n_fft//2+1:raise ValueError('Invalid spectral configuration')
    count=1+max(0,math.ceil((len(x)-n_fft)/hop))
    if count>10000:raise ValueError('Spectrogram exceeds 10000 frames')
    padded=np.pad(x,(0,max(0,(count-1)*hop+n_fft-len(x))))
    # Bounded incremental summaries avoid retaining all STFT frames.
    total=np.zeros(bands);squares=np.zeros(bands);groups=np.array_split(np.arange(n_fft//2+1),bands)
    window=np.hanning(n_fft)
    for i in range(count):
        power=np.abs(np.fft.rfft(padded[i*hop:i*hop+n_fft]*window))**2
        row=np.array([np.log1p(power[group].mean()) for group in groups])
        total+=row;squares+=row*row
    mean=total/count;std=np.sqrt(np.maximum(0,squares/count-mean*mean))
    rms=float(np.sqrt(np.mean(x.astype(float)**2)));zcr=float(np.mean((x[1:]>=0)!=(x[:-1]>=0))) if len(x)>1 else 0.
    return np.concatenate([mean,std,[rms,zcr]]).astype(np.float32)


def mono(waveform):
    """Explicit channel-mean downmix, returning an owned waveform."""
    x=_waveform(waveform)
    return Waveform(x.mean(axis=0,keepdims=True).copy(),waveform.sample_rate)


def save_wav(waveform,path):
    """Create a new16-bit PCM WAV; no overwrite or implicit amplitude clipping."""
    np=require('numpy','audio');x=_waveform(waveform)
    path=Path(path)
    with path.open('xb') as stream:
        with wave.open(stream,'wb') as writer:
            writer.setnchannels(x.shape[0]);writer.setsampwidth(2);writer.setframerate(waveform.sample_rate)
            writer.writeframes(np.rint(x.T*32767).astype('<i2').tobytes())
    return path


def load_asr(path,*,kind='ctc',max_model_bytes=1024**3):
    """Load an existing local CTC/seq2seq model and processor; never download."""
    from nil._speech_models import Recognizer
    return Recognizer(path,kind=kind,max_model_bytes=max_model_bytes)


def load_tts(path,*,max_model_bytes=1024**3):
    """Load an existing local VITS model/tokenizer; native weights remain explicit."""
    from nil._speech_models import Synthesizer
    return Synthesizer(path,max_model_bytes=max_model_bytes)


def word_error_rate(reference,hypothesis):
    """Case-folded Unicode word WER; punctuation removed, insertions may exceed1."""
    import re
    if not isinstance(reference,str) or not isinstance(hypothesis,str):raise TypeError('Text strings required')
    first=re.findall(r'\w+',reference.casefold());second=re.findall(r'\w+',hypothesis.casefold())
    if not first or max(len(first),len(second))>2000:raise ValueError('Nonempty bounded reference required')
    previous=list(range(len(second)+1))
    for i,a in enumerate(first,1):
        row=[i]
        for j,b in enumerate(second,1):row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=row
    return previous[-1]/len(first)


def stream(recognizer,**options):
    """Create bounded fixed-window recognition, explicit push/flush and snapshots."""
    from nil._speech_stream import Stream
    return Stream(recognizer,**options)


def load_stream(path,*,recognizer):
    """Resume pending samples and absolute offsets with the same recognizer."""
    from nil._speech_stream import Stream
    return Stream.load(path,recognizer=recognizer)
