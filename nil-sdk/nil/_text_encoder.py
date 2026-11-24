"""Local-only mean-pooling Transformer encoder for explicit semantic retrieval."""
import hashlib
from nil.runtime import require
from nil._local_models import identity,publish


class TextEncoder:
    def __init__(self,path,*,max_length=128,batch_size=16,max_model_bytes=1024**3):
        if type(max_length) is not int or not 1<=max_length<=512 or type(batch_size) is not int or not 1<=batch_size<=64:raise ValueError('Invalid encoder sequence/batch budget')
        root,model_id,_=identity(path,max_bytes=max_model_bytes)
        self.fingerprint=hashlib.sha256(f"{model_id}:masked_mean_l2_v1:{max_length}".encode()).hexdigest()
        hf=require('transformers','generative')
        self.tokenizer=hf.AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
        self.model=hf.AutoModel.from_pretrained(root,local_files_only=True,trust_remote_code=False)
        self.max_length=max_length;self.batch_size=batch_size

    def encode(self,texts):
        torch=require('torch','torch');np=require('numpy','audio')
        if not isinstance(texts,(list,tuple)) or not 1<=len(texts)<=10000 or any(not isinstance(s,str) or not s.strip() or len(s)>10000 for s in texts):raise ValueError('Bounded nonempty text list required')
        if any(p.device.type!='cpu' for p in self.model.parameters()):raise ValueError('Current text encoder acceptance requires CPU')
        result=[];previous=self.model.training;self.model.eval()
        try:
            with torch.inference_mode():
                for start in range(0,len(texts),self.batch_size):
                    inputs=self.tokenizer(texts[start:start+self.batch_size],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt')
                    values=self.model(**inputs).last_hidden_state
                    mask=inputs['attention_mask'].unsqueeze(-1).to(values.dtype)
                    pooled=(values*mask).sum(1)/mask.sum(1).clamp_min(1)
                    if not torch.isfinite(pooled).all() or (pooled.norm(dim=1)==0).any():raise ValueError('Invalid text embeddings')
                    result.append(torch.nn.functional.normalize(pooled,dim=1).cpu().numpy())
        finally:self.model.train(previous)
        return np.concatenate(result)

    def save(self,path):return publish(self.model,self.tokenizer,path,kind='mean-pooled-text-encoder')
