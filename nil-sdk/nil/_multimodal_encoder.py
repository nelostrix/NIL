"""Local-only CLIP encoder; weights and Transformers are explicit user inputs."""
from nil.runtime import require
from nil._local_models import identity,publish


class Encoder:
    def __init__(self,path,*,max_model_bytes=1024**3):
        root,self.fingerprint,_=identity(path,max_bytes=max_model_bytes)
        hf=require('transformers','generative')
        image_processor=hf.CLIPImageProcessor.from_pretrained(root,local_files_only=True)
        tokenizer=hf.AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False,use_fast=True)
        self.processor=hf.CLIPProcessor(image_processor=image_processor,tokenizer=tokenizer)
        self.model=hf.CLIPModel.from_pretrained(root,local_files_only=True,trust_remote_code=False)

    def _encode(self,kind,values):
        torch=require('torch','torch')
        if not isinstance(values,(list,tuple)) or not 1<=len(values)<=32:raise ValueError('Require 1..32 items per batch')
        if any(p.device.type!='cpu' for p in self.model.parameters()):raise ValueError('Current CLIP adapter requires CPU')
        if kind=='text' and any(not isinstance(s,str) or not s.strip() or len(s)>10000 for s in values):raise ValueError('Invalid text')
        if kind=='images':
            image=require('PIL.Image','vision')
            if any(not isinstance(v,image.Image) or v.width*v.height>4*1024*1024 for v in values):raise ValueError('Require bounded PIL images')
        arguments=dict(text=values,padding=True,truncation=True,max_length=self.model.config.text_config.max_position_embeddings) if kind=='text' else dict(images=values)
        inputs=self.processor(**arguments,return_tensors='pt')
        inputs.pop('token_type_ids',None)
        previous=self.model.training;self.model.eval()
        try:
            with torch.inference_mode():
                result=(self.model.get_text_features if kind=='text' else self.model.get_image_features)(**inputs)
                if not torch.isfinite(result).all() or (result.norm(dim=1)==0).any():raise ValueError('Invalid projected embeddings')
                return torch.nn.functional.normalize(result,dim=1).cpu().numpy()
        finally:self.model.train(previous)

    def encode_text(self,texts):return self._encode('text',texts)
    def encode_images(self,images):return self._encode('images',images)
    def save(self,path):return publish(self.model,self.processor,path,kind='clip')
