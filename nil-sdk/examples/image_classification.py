"""Train/evaluate local images, save preprocessing/labels, and reload predictions.

Omit --train/--valid for an eight-image wiring demo (not a quality benchmark).
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def exercise(output, *, train_root=None, valid_root=None, epochs=15, size=16, width=8):
    import nil
    import torch
    from PIL import Image
    started=time.monotonic()
    torch.set_num_threads(1);nil.runtime.seed(4)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if train_root is None:
        train_root=output/'demo-images'
        for label,color in [('red',(255,0,0)),('blue',(0,0,255))]:
            folder=train_root/label;folder.mkdir(parents=True,exist_ok=True)
            for i in range(4):Image.new('RGB',(8,8),color).save(folder/f'{i}.png')
        valid_root=train_root
    if valid_root is None:
        raise ValueError('Provide a separate --valid folder with --train')
    transform=nil.vision.ImageTransform((size,size))
    train=nil.vision.image_folders(train_root,transform=transform)
    valid=nil.vision.image_folders(valid_root,classes=train.classes,transform=transform)
    model=nil.model('nelo_vision_mini',num_classes=len(train.classes),width=width)
    before=nil.evaluate(model,valid,task='classification',device_name='cpu')
    result=nil.train(model,train,task='classification',validation=valid,epochs=epochs,
                     batch_size=32,learning_rate=.01,device='cpu',output_dir=output/'runs')
    after=nil.evaluate(result,valid,task='classification',device_name='cpu')
    manifest_file=output/'image-manifest.json'
    manifest_file.write_text(json.dumps(train.manifest(),indent=2)+'\n')
    manifest=json.loads(manifest_file.read_text())
    restored=nil.model('nelo_vision_mini',num_classes=len(manifest['classes']),width=width)
    nil.training.checkpoint.load(result.checkpoint,restored)
    paths=[path for path,label in valid.samples]
    expected=list(nil.vision.classify(result,paths,manifest=manifest))
    actual=list(nil.vision.classify(restored,paths,manifest=manifest))
    if actual!=expected:
        raise AssertionError('Checkpoint/manifest inference roundtrip changed predictions')
    if not after['loss']<before['loss']:
        raise AssertionError('Learning check did not reduce held-out loss')
    digest=hashlib.sha256()
    for split,dataset,root in [('train',train,Path(train_root)),('valid',valid,Path(valid_root))]:
        for path,label in dataset.samples:
            digest.update(json.dumps([split,str(path.relative_to(root)),label],sort_keys=True).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    report={'schema_version':1,'status':'passed','task':'image_classification',
            'scope':'CPU image pipeline and checkpoint acceptance; dataset quality is scoped to these files',
            'train_samples':len(train),'validation_samples':len(valid),'dataset_sha256':digest.hexdigest(),
            'classes':list(train.classes),'before':before,'after':after,'exact_checkpoint_reload':True,
            'model':{'name':'nelo_vision_mini','width':width},'epochs':epochs,'transform':transform.to_dict(),
            'seconds':time.monotonic()-started,'checkpoint':str(result.checkpoint)}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path);parser.add_argument('--valid',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=15);parser.add_argument('--size',type=int,default=16)
    parser.add_argument('--width',type=int,default=8)
    args=parser.parse_args()
    print(json.dumps(exercise(args.output,train_root=args.train,valid_root=args.valid,
                             epochs=args.epochs,size=args.size,width=args.width),indent=2))

if __name__=='__main__':main()
