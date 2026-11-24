"""Native HF speech adapters; imported only by explicit local model loading."""
from nil.runtime import require
from nil._local_models import identity,publish
from nil.audio import Waveform,_waveform


def _cpu(model):
    if any(p.device.type!='cpu' for p in model.parameters()):raise ValueError('This accepted speech adapter currently requires CPU model parameters')


class Recognizer:
    def __init__(self,path,*,kind='ctc',max_model_bytes=1024**3):
        if kind not in {'ctc','seq2seq'}:raise ValueError('Choose ctc or seq2seq recognition')
        root,self.fingerprint,_=identity(path,max_bytes=max_model_bytes)
        hf=require('transformers','generative')
        self.processor=hf.AutoProcessor.from_pretrained(root,local_files_only=True,trust_remote_code=False)
        factory=hf.AutoModelForCTC if kind=='ctc' else hf.AutoModelForSpeechSeq2Seq
        self.model=factory.from_pretrained(root,local_files_only=True,trust_remote_code=False)
        self.kind=kind
        self.sample_rate=int(self.processor.feature_extractor.sampling_rate)

    def transcribe(self,waveform,*,max_new_tokens=128):
        torch=require('torch','torch');samples=_waveform(waveform);_cpu(self.model)
        if samples.shape[0]!=1 or waveform.sample_rate!=self.sample_rate:raise ValueError('ASR requires explicit mono conversion and the processor sample rate')
        if waveform.seconds>30:raise ValueError('ASR accepts at most30 seconds; segment explicitly')
        if type(max_new_tokens) is not int or not 1<=max_new_tokens<=512:raise ValueError('Invalid decoding token budget')
        inputs=self.processor(samples[0],sampling_rate=self.sample_rate,return_tensors='pt',padding=True)
        previous=self.model.training;self.model.eval()
        try:
            with torch.inference_mode():
                if self.kind=='ctc':
                    logits=self.model(**inputs).logits
                    if not torch.isfinite(logits).all():raise ValueError('Nonfinite recognition logits')
                    tokens=logits.argmax(-1)
                else:tokens=self.model.generate(**inputs,max_new_tokens=max_new_tokens)
                text=self.processor.batch_decode(tokens,skip_special_tokens=True)[0]
        finally:self.model.train(previous)
        return dict(text=text,seconds=waveform.seconds,sample_rate=self.sample_rate,model_fingerprint=self.fingerprint)

    def save(self,path):return publish(self.model,self.processor,path,kind='asr-'+self.kind)


class Synthesizer:
    def __init__(self,path,*,max_model_bytes=1024**3):
        root,self.fingerprint,_=identity(path,max_bytes=max_model_bytes)
        hf=require('transformers','generative')
        self.processor=hf.AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
        self.model=hf.VitsModel.from_pretrained(root,local_files_only=True,trust_remote_code=False)
        self.sample_rate=int(self.model.config.sampling_rate)

    def synthesize(self,text,*,seed=0,max_seconds=30.):
        torch=require('torch','torch');np=require('numpy','audio');_cpu(self.model)
        if not isinstance(text,str) or not text.strip() or len(text)>500:raise ValueError('Provide1–500 characters of text')
        if type(seed) is not int or not 0<=seed<2**32 or type(max_seconds) not in (int,float) or not 0<max_seconds<=60:raise ValueError('Invalid speech budget/seed')
        inputs=self.processor(text,return_tensors='pt')
        if inputs['input_ids'].numel()>256:raise ValueError('Text exceeds256 tokens')
        previous=self.model.training;self.model.eval()
        try:
            with torch.inference_mode(),torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                samples=self.model(**inputs).waveform.detach().cpu().float().numpy()
        finally:self.model.train(previous)
        if samples.ndim!=2 or samples.shape[0]!=1 or not 1<=samples.shape[1]<=max_seconds*self.sample_rate or not np.isfinite(samples).all() or (np.abs(samples)>1).any():raise ValueError('Synthesized waveform exceeds duration/amplitude contract')
        return Waveform(samples,self.sample_rate)

    def save(self,path):return publish(self.model,self.processor,path,kind='tts-vits')
