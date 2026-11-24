"""Two-view self-supervised training and explicit few-shot held-out evaluation."""
import argparse
import json
from pathlib import Path


def exercise(output):
    import nil
    import torch
    torch.set_num_threads(1);torch.manual_seed(7)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    labels=torch.arange(96)%3
    x=torch.nn.functional.one_hot(labels,3).float()*2+.1*torch.randn(96,3)
    view=x+.05*torch.randn_like(x)
    encoder=nil.nn.mlp(3,8,hidden=(16,))
    class Objective:
        def __call__(self,model,batch):return nil.learning.contrastive_loss(model(batch[0]),model(batch[1]))
    task=Objective();before=task(encoder,(x[:32],view[:32])).item()
    result=nil.train(encoder,nil.data.tensors(x,view),task=task,epochs=25,batch_size=32,
                     learning_rate=.01,device='cpu',output_dir=output/'encoder')
    after=task(result.model,(x[:32],view[:32])).item()
    if not after<before:raise AssertionError('Self-supervised objective did not improve')
    test_labels=torch.arange(48)%3
    test_x=torch.nn.functional.one_hot(test_labels,3).float()*2+.1*torch.randn(48,3)
    support=nil.predict(result,x[:9],device_name='cpu').numpy();queries=nil.predict(result,test_x,device_name='cpu').numpy()
    prediction=nil.learning.prototypes(support,labels[:9].numpy(),queries)
    accuracy=float((prediction['labels']==test_labels.numpy()).mean())
    if accuracy<.8:raise AssertionError('Held-out prototype quality floor failed')
    restored=nil.nn.mlp(3,8,hidden=(16,));nil.training.checkpoint.load(result.checkpoint,restored)
    torch.testing.assert_close(nil.predict(restored,test_x,device_name='cpu'),nil.predict(result,test_x,device_name='cpu'),rtol=0,atol=0)
    nil.data.snapshot([dict(embedding=row.tolist(),label=int(label)) for row,label in zip(support,labels[:9])],output/'support')
    recovered=nil.data.load_snapshot(output/'support')
    loaded=nil.learning.prototypes([r['embedding'] for r in recovered],[r['label'] for r in recovered],queries)
    assert (loaded['labels']==prediction['labels']).all()
    report=dict(status='passed',scope='Synthetic two-view representation learning and nine-example prototype support; not general few-shot/continual-learning quality',
        before_contrastive_loss=before,after_contrastive_loss=after,support_examples=9,held_out_examples=48,
        held_out_accuracy=accuracy,exact_checkpoint_reload=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(exercise(args.output),indent=2))
