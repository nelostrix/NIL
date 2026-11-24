"""CPU research plumbing: untied/shared attention, offline distillation and reload.

Synthetic classification only; not a language-model or intelligence benchmark.
Requires explicitly installed nil-sdk[torch]. No model/data downloads.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def build_model(*, shared=True, depth=4, width=16, heads=2, vocab=16, length=6):
    import torch
    if type(shared) is not bool:
        raise ValueError('shared must be boolean')
    for name,value,maximum in [('depth',depth,16),('width',width,128),('heads',heads,16),('vocab',vocab,256),('length',length,64)]:
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f'Invalid bounded {name}')
    if width % heads:
        raise ValueError('width must be divisible by heads')
    config=dict(shared=shared,depth=depth,width=width,heads=heads,vocab=vocab,length=length)

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding=torch.nn.Embedding(vocab,width)
            self.position=torch.nn.Parameter(torch.zeros(1,length,width))
            self.blocks=torch.nn.ModuleList([
                torch.nn.TransformerEncoderLayer(width,heads,dim_feedforward=2*width,
                    dropout=0.,batch_first=True) for _ in range(1 if shared else depth)])
            self.head=torch.nn.Linear(width,2)

        def extra_repr(self):
            # Reuse depth has no additional state-dict keys; preserve it in the
            # existing NIL structural checkpoint preflight before any weight load.
            return json.dumps(config,sort_keys=True)

        def forward(self,tokens):
            if tokens.ndim != 2 or tokens.shape[1] != length:
                raise ValueError('Expected [batch, configured sequence length] token IDs')
            values=self.embedding(tokens)+self.position
            for index in range(depth):
                values=self.blocks[0 if shared else index](values)
            return self.head(values[:,0])

    return Classifier()


class DistillationStep:
    """alpha*CE + (1-alpha)*T^2*KL(teacher || student), mean over examples.

    Teacher logits are frozen, offline dataset entries. This is not an online
    teacher lifecycle, language-token distillation or learned adaptive halting.
    """
    def __init__(self, *, alpha=.5, temperature=2.):
        if type(alpha) not in (int,float) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('alpha must be in [0,1]')
        if type(temperature) not in (int,float) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be positive and finite')
        self.alpha=alpha;self.temperature=temperature

    def normalization_count(self,batch):
        return float(batch[1].numel())

    def __call__(self,model,batch):
        import torch
        inputs,labels,teacher_logits=batch
        student=model(inputs)
        teacher=teacher_logits.detach()
        if student.shape != teacher.shape or student.ndim != 2 or labels.shape != student.shape[:1]:
            raise ValueError('Expected aligned [batch, classes] logits and [batch] labels')
        if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
            raise ValueError('Distillation logits must be finite')
        t=self.temperature
        hard=torch.nn.functional.cross_entropy(student,labels)
        soft=torch.nn.functional.kl_div((student/t).log_softmax(-1),(teacher/t).softmax(-1),reduction='batchmean')*t*t
        return self.alpha*hard+(1-self.alpha)*soft


def exercise(output, *, seed=7, epochs=12):
    import nil
    import torch
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be in [0, 2**32)')
    if type(epochs) is not int or not 1 <= epochs <= 100:
        raise ValueError('epochs must be from 1 through 100')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(1)
    generator=torch.Generator().manual_seed(seed)
    train_x=torch.randint(0,16,(96,6),generator=generator)
    valid_x=torch.randint(0,16,(48,6),generator=generator)
    assert not set(map(tuple,train_x.tolist())) & set(map(tuple,valid_x.tolist()))
    train_y=train_x[:,0] % 2;valid_y=valid_x[:,0] % 2
    train=nil.data.tensors(train_x,train_y);valid=nil.data.tensors(valid_x,valid_y)
    digest=hashlib.sha256()
    for values in (train_x,train_y,valid_x,valid_y):digest.update(values.numpy().tobytes())
    counts={};rows={};teacher_train=teacher_valid=None
    for name,shared,distilled in [('untied_baseline',False,False),('shared_supervised',True,False),('shared_distilled',True,True)]:
        nil.runtime.seed(seed);model=build_model(shared=shared)
        counts[name]=sum(p.numel() for p in model.parameters())
        before=nil.evaluate(model,valid,task='classification',device_name='cpu')
        task=DistillationStep() if distilled else 'classification'
        training=nil.data.tensors(train_x,train_y,teacher_train) if distilled else train
        validation=nil.data.tensors(valid_x,valid_y,teacher_valid) if distilled else valid
        begin=time.monotonic()
        result=nil.train(model,training,task=task,validation=validation,epochs=epochs,
                         batch_size=16,learning_rate=.01,seed=seed,device='cpu',output_dir=output/name)
        after=nil.evaluate(result,valid,task='classification',device_name='cpu')
        if not after['loss'] < before['loss']:
            raise AssertionError(f'{name}: held-out loss did not improve')
        restored=build_model(shared=shared)
        nil.training.checkpoint.load(result.checkpoint,restored)
        expected=nil.predict(result,valid_x,device_name='cpu')
        actual=nil.predict(restored,valid_x,device_name='cpu')
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        if name=='untied_baseline':
            teacher_train=nil.predict(result,train_x,device_name='cpu').detach()
            teacher_valid=expected.detach()
        rows[name]={'before':before,'after':after,'training_seconds':time.monotonic()-begin,
                    'exact_reload':True,'architecture':{'shared':shared,'depth':4,'width':16,'heads':2,'vocab':16,'length':6}}
    assert counts['shared_supervised']==counts['shared_distilled'] < counts['untied_baseline']
    report={'schema_version':1,'status':'passed','scope':'Synthetic CPU integration smoke: fixed-depth shared attention and offline classification-logit distillation; no intelligence-equivalence claim',
            'seed':seed,'epochs':epochs,'train_examples':96,'held_out_examples':48,
            'dataset_sha256':digest.hexdigest(),'parameter_counts':counts,'runs':rows,
            'seconds':time.monotonic()-started,'adaptive_halting':False,'compute_matched':False,
            'teacher_cost_included_in_total_seconds':True,'large_model_readiness':False}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--epochs',type=int,default=12)
    args=parser.parse_args()
    print(json.dumps(exercise(args.output,seed=args.seed,epochs=args.epochs),indent=2))

if __name__=='__main__':main()
