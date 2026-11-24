"""Incremental fixed-window recognition with explicit offsets and bounded buffering."""
import math
from nil.runtime import require
from nil.audio import Waveform,_waveform
from nil._workflow_io import save as _save,load as _load


class Stream:
    """Nonoverlapping fixed audio windows, not a stateful neural streaming decoder.

    Call push(Waveform) for any block size; flush() decodes the final partial
    window and closes the stream. Words may be split at boundaries. There is no
    inferred word alignment, automatic resampling, capture thread or download.
    """
    def __init__(self,recognizer,*,window_seconds=5.,min_tail_seconds=.1,max_windows_per_push=64):
        if not callable(getattr(recognizer,'transcribe',None)) or not isinstance(getattr(recognizer,'fingerprint',None),str) or not recognizer.fingerprint or type(getattr(recognizer,'sample_rate',None)) is not int or not 1<=recognizer.sample_rate<=192000:raise ValueError('Require a recognizer with identity and sample rate')
        if type(window_seconds) not in (int,float) or not math.isfinite(window_seconds) or not .1<=window_seconds<=30 or type(min_tail_seconds) not in (int,float) or not math.isfinite(min_tail_seconds) or not 0<=min_tail_seconds<=window_seconds or type(max_windows_per_push) is not int or not 1<=max_windows_per_push<=256:raise ValueError('Invalid window/tail budget')
        self.recognizer=recognizer;self.fingerprint=recognizer.fingerprint;self.sample_rate=recognizer.sample_rate
        self.window_samples=round(window_seconds*self.sample_rate);self.min_tail_samples=round(min_tail_seconds*self.sample_rate)
        if self.window_samples<1:raise ValueError('Window must contain at least one sample')
        self.max_windows_per_push=max_windows_per_push;self.offset=0;self.closed=False
        np=require('numpy','audio');self._pending=np.empty((1,0),dtype=np.float32)

    def _check(self):
        if self.recognizer.fingerprint!=self.fingerprint or self.recognizer.sample_rate!=self.sample_rate:raise ValueError('Recognizer identity/rate changed')
        if self.closed:raise ValueError('Stream is closed')

    def _decode(self,samples,offset):
        value=self.recognizer.transcribe(Waveform(samples.copy(),self.sample_rate))
        if not isinstance(value,dict) or not isinstance(value.get('text'),str) or len(value['text'])>100000:raise ValueError('Invalid recognition output')
        return dict(text=value['text'],start_sample=offset,end_sample=offset+samples.shape[1],
                    start_seconds=offset/self.sample_rate,end_seconds=(offset+samples.shape[1])/self.sample_rate,
                    model_fingerprint=self.fingerprint)

    def push(self,waveform):
        self._check();samples=_waveform(waveform);np=require('numpy','audio')
        if waveform.sample_rate!=self.sample_rate or samples.shape[0]!=1:raise ValueError('Explicit mono/rate conversion required')
        if self.offset+self._pending.shape[1]+samples.shape[1]>2**63-1:raise ValueError('Stream sample counter exhausted')
        count=(self._pending.shape[1]+samples.shape[1])//self.window_samples
        if count>self.max_windows_per_push:raise ValueError('Push exceeds decode-window budget; use smaller blocks')
        combined=np.concatenate([self._pending,samples],axis=1);records=[]
        for number in range(count):
            start=number*self.window_samples
            records.append(self._decode(combined[:,start:start+self.window_samples],self.offset+start))
        # Commit only after every decode succeeds. Retrying on model failure may
        # call the pure recognizer again; no external tool effects are supported.
        self.offset+=count*self.window_samples;self._pending=combined[:,count*self.window_samples:].copy()
        return records

    def flush(self):
        self._check();records=[];count=self._pending.shape[1]
        if count and count<self.min_tail_samples:raise ValueError('Final tail is below minimum duration; append more audio')
        if count:records=[self._decode(self._pending,self.offset)]
        self.offset+=count;self._pending=self._pending[:,:0].copy();self.closed=True;return records

    def save(self,path):
        self._check_identity()
        return _save(path,'speech-stream',dict(fingerprint=self.fingerprint,sample_rate=self.sample_rate,
            window_samples=self.window_samples,min_tail_samples=self.min_tail_samples,max_windows_per_push=self.max_windows_per_push,
            offset=self.offset,closed=self.closed,pending=self._pending.tolist()))

    def _check_identity(self):
        if self.recognizer.fingerprint!=self.fingerprint or self.recognizer.sample_rate!=self.sample_rate:raise ValueError('Recognizer identity/rate changed')

    @classmethod
    def load(cls,path,*,recognizer):
        np=require('numpy','audio');r=_load(path,'speech-stream')
        if not isinstance(r,dict) or set(r)!={'fingerprint','sample_rate','window_samples','min_tail_samples','max_windows_per_push','offset','closed','pending'}:raise ValueError('Invalid speech stream schema')
        if r['fingerprint']!=getattr(recognizer,'fingerprint',None) or r['sample_rate']!=getattr(recognizer,'sample_rate',None):raise ValueError('Recognizer identity/rate mismatch')
        rate=r['sample_rate']
        if type(r['window_samples']) is not int or type(r['min_tail_samples']) is not int:raise ValueError('Invalid sample budgets')
        stream=cls(recognizer,window_seconds=r['window_samples']/rate,min_tail_seconds=r['min_tail_samples']/rate,max_windows_per_push=r['max_windows_per_push'])
        pending=np.asarray(r['pending'],dtype=np.float32)
        if pending.ndim!=2 or pending.shape[0]!=1 or pending.shape[1]>=stream.window_samples or not np.isfinite(pending).all() or (abs(pending)>1).any() or type(r['offset']) is not int or not 0<=r['offset']<=2**63-1 or type(r['closed']) is not bool or r['offset']+pending.shape[1]>2**63-1 or r['closed'] and pending.shape[1] or not r['closed'] and r['offset']%stream.window_samples:raise ValueError('Invalid stream continuation state')
        stream.offset=r['offset'];stream.closed=r['closed'];stream._pending=pending.copy();return stream
