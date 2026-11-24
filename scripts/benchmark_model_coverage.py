"""Offline CPU workflow evidence on bundled digits and separate local text files.

Prepare digits using an explicit classical environment, then run in an explicit
Torch/generative/ONNX environment. No remote datasets/models are downloaded.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tempfile
import time


def prepare(path):
    import numpy as np
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    digits=load_digits();train,test=train_test_split(np.arange(len(digits.target)),test_size=.2,stratify=digits.target,random_state=7)
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,x_train=(digits.data[train]/16).astype('float32'),y_train=digits.target[train],
        x_test=(digits.data[test]/16).astype('float32'),y_test=digits.target[test])
    path.with_suffix('.json').write_text(json.dumps({'source':'scikit-learn bundled load_digits, pixel values divided by16',
        'split':'stratified80/20, seed7','sklearn':version('scikit-learn'),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2)+'\n')
    print(path)


def digits(path,directory):
    import numpy as np
    import torch
    import nil
    data=np.load(path,allow_pickle=False);source=json.loads(path.with_suffix('.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Digits fixture digest differs')
    x=torch.as_tensor(data['x_train']);y=torch.as_tensor(data['y_train']).long()
    test=torch.as_tensor(data['x_test']);labels=torch.as_tensor(data['y_test']).long()
    torch.manual_seed(7);model=nil.nn.mlp(64,10,hidden=(32,))
    learned=nil.train(model,nil.data.tensors(x,y),task='classification',epochs=20,batch_size=128,
        learning_rate=.005,device='cpu',seed=7,output_dir=directory/'digits')
    learned.model.eval()
    with torch.inference_mode():
        reference=learned.model(test);accuracy=float((reference.argmax(1)==labels).float().mean())
    quantized=nil.optimization.quantize(learned.model)
    with torch.inference_mode():quantized_accuracy=float((quantized(test).argmax(1)==labels).float().mean())
    exported=nil.optimization.export_onnx(learned.model,test[:2],directory/'digits.onnx',input_names=['features'],output_names=['logits'],
        dynamic_axes={'features':{0:'batch'},'logits':{0:'batch'}})
    outputs=nil.inference.OnnxModel(exported).predict(test.numpy())[0]
    np.testing.assert_allclose(outputs,reference.numpy(),rtol=1e-4,atol=1e-5)
    passed=accuracy>=.9 and quantized_accuracy>=accuracy-.03
    return {'status':'passed' if passed else 'failed','source':source,'train_rows':len(x),'heldout_rows':len(test),'epochs':20,
        'seed':7,'accuracy':accuracy,'dynamic_int8_accuracy':quantized_accuracy,'onnx_max_abs_error':float(np.max(np.abs(outputs-reference.numpy()))),
        'parameters':sum(p.numel() for p in model.parameters()),'scope':'Real bundled digit images, one fixed split/seed, small MLP; not general vision/pretrained or quantization coverage'}


def language(directory):
    import torch
    import nil
    root=Path(__file__).resolve().parents[1];train_files=[root/'README.md',root/'docs/CLASSICAL_ML.md'];heldout_file=root/'docs/INSTALLATION.md'
    def paragraphs(paths):return [text.strip() for path in paths for text in path.read_text().split('\n\n') if len(text.strip())>=20][:160]
    training=paragraphs(train_files);heldout=paragraphs([heldout_file]);torch.manual_seed(7)
    tokenizer=nil.generative.tokenizer_from_texts(training,vocab_size=512)
    train=nil.generative.text_dataset(training,tokenizer,max_length=64);test=nil.generative.text_dataset(heldout,tokenizer,max_length=64)
    model=nil.generative.causal_lm(len(tokenizer),context_length=64,width=32,layers=2,heads=2)
    before=nil.generative.perplexity(model,test,device_name='cpu')
    trained=nil.train(model,train,task='causal_lm',epochs=8,batch_size=8,learning_rate=.003,device='cpu',seed=7,output_dir=directory/'language-base')
    after=nil.generative.perplexity(trained.model,test,device_name='cpu')
    adapted=nil.generative.lora(trained.model,rank=4,alpha=8,target_modules=['c_attn'])
    counts=nil.nn.parameters(adapted);frozen={name:p.detach().clone() for name,p in adapted.named_parameters() if not p.requires_grad}
    result=nil.train(adapted,train,task='causal_lm',epochs=3,batch_size=8,learning_rate=.001,device='cpu',seed=7,output_dir=directory/'language-adapter')
    for name,p in result.model.named_parameters():
        if name in frozen:torch.testing.assert_close(p,frozen[name],rtol=0,atol=0)
    score=nil.generative.perplexity(result.model,test,device_name='cpu');merged=result.model.merge_and_unload()
    destination=directory/'language-merged';nil.generative.save(merged,tokenizer,destination)
    restored,_=nil.generative.load_causal_lm(destination)
    loaded=nil.generative.perplexity(restored,test,device_name='cpu')
    if abs(score['nll']-loaded['nll'])>1e-5:raise ValueError('Merged local model did not reproduce held-out NLL')
    passed=after['nll']<before['nll'] and 0<counts['trainable']<counts['total']
    return {'status':'passed' if passed else 'failed','training_paragraphs':len(train),'heldout_paragraphs':len(test),
        'files':{path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in [*train_files,heldout_file]},
        'tokenizer':'Byte BPE fitted on training files only,512 tokens, context64','base_epochs':8,'adapter_epochs':3,
        'initial':before,'base_trained':after,'adapted':score,'merged_reloaded':loaded,'lora_parameters':counts,
        'scope':'Small local GPT2-style model from scratch on separate repository documents; not downloaded pretrained checkpoints, large-model quality or general language benchmarks'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-digits',type=Path)
    parser.add_argument('--digits-data',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.prepare_digits:prepare(args.prepare_digits);return 0
    if not args.digits_data or not args.output:parser.error('--digits-data and --output required')
    import torch
    torch.set_num_threads(1);started=time.monotonic()
    report={'schema_version':1,'status':'failed','python':platform.python_version(),'platform':platform.platform(),
        'versions':{name:version(name) for name in ('torch','numpy','transformers','peft','onnx','onnxruntime')}}
    with tempfile.TemporaryDirectory(prefix='nil-model-coverage-') as temporary:
        report['digits']=digits(args.digits_data,Path(temporary));report['language']=language(Path(temporary))
    report['seconds']=time.monotonic()-started;report['status']='passed' if all(report[key]['status']=='passed' for key in ('digits','language')) else 'failed'
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'seconds':report['seconds'],'digits_accuracy':report['digits']['accuracy'],
        'initial_nll':report['language']['initial']['nll'],'trained_nll':report['language']['base_trained']['nll']}));return 0 if report['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
