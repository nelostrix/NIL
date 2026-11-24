"""Explicit, pinned, byte-budgeted public research fixture preparation.

Never called by NIL installation/import. Weights stay outside the repository.
MMS-TTS research fixture is CC-BY-NC-4.0, not a commercial-use recommendation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

MODELS={
 'asr':('facebook/wav2vec2-base-960h','22aad52d435eb6dbaf354bdad9b0da84ce7d6156'),
 'tts':('facebook/mms-tts-eng','c71de0fe7204c83f1c10820a7d696d0b450048ba'),
 'text':('sentence-transformers/paraphrase-MiniLM-L3-v2','4ca70771034acceecb2e72475f72050fcdde4ddc'),
 'vision':('wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M','a2a8c6eaa2549ad66eb7c31b85022bf58273a26c')}
FILES={'config.json','feature_extractor_config.json','preprocessor_config.json','model.safetensors',
       'special_tokens_map.json','tokenizer_config.json','tokenizer.json','vocab.json','vocab.txt','merges.txt'}


def prepare(root,selected,max_bytes):
    from huggingface_hub import HfApi
    api=HfApi(token=False);items=[];total=0
    for key in selected:
        name,revision=MODELS[key];info=api.model_info(name,revision=revision,files_metadata=True)
        if info.sha!=revision:raise ValueError('Pinned model revision mismatch')
        files=[f for f in info.siblings if f.rfilename in FILES]
        if not any(f.rfilename=='model.safetensors' for f in files):raise ValueError('Safe weight file missing')
        total+=sum(f.size for f in files)
        items.append((key,name,revision,getattr(info.card_data,'license',None),files))
    if total>max_bytes:raise ValueError('Selected models exceed explicit total download budget')
    root.mkdir(parents=True,exist_ok=True)
    def fetch(item):
        key,name,revision,license,files=item;directory=root/key;directory.mkdir(exist_ok=True);records=[]
        for f in files:
            target=directory/f.rfilename;expected=getattr(f.lfs,'sha256',None) if f.lfs else None
            def verify(path):
                if path.stat().st_size!=f.size:return False
                digest=hashlib.sha256();git=hashlib.sha1(b'blob '+str(f.size).encode()+b'\0')
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block);git.update(block)
                return (digest.hexdigest()==expected if expected else git.hexdigest()==f.blob_id)
            if not target.exists() or not verify(target):
                temporary=target.with_suffix(target.suffix+'.partial')
                try:
                    with urlopen(f'https://huggingface.co/{name}/resolve/{revision}/{f.rfilename}',timeout=60) as response,temporary.open('wb') as stream:
                        remaining=f.size+1
                        while remaining:
                            block=response.read(min(1024*1024,remaining))
                            if not block:break
                            stream.write(block);remaining-=len(block)
                    if not verify(temporary):raise ValueError('Downloaded file failed size/hash verification: '+key+'/'+f.rfilename)
                    temporary.replace(target)
                finally:temporary.unlink(missing_ok=True)
            digest=hashlib.sha256(target.read_bytes()).hexdigest() if f.size<1024*1024 else expected
            records.append(dict(file=f.rfilename,bytes=f.size,sha256=digest))
        manifest=dict(model=name,revision=revision,license=license,files=records)
        (directory/'.download-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return dict(key=key,model=name,revision=revision,license=license,bytes=sum(f.size for f in files))
    return list(ThreadPoolExecutor(2).map(fetch,items))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS));parser.add_argument('--max-bytes',type=int,default=750*1024*1024)
    args=parser.parse_args();print(json.dumps(prepare(args.output,args.models,args.max_bytes),indent=2))
