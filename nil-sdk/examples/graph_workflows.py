"""CPU GCN classification with explicit train/test labels and exact checkpoint reload."""
import argparse
import json
from pathlib import Path


def exercise(output, *, graph_file=None):
    import nil
    import torch
    torch.set_num_threads(1);torch.manual_seed(7)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if graph_file is None:
        labels=torch.arange(40)%2
        features=torch.stack([labels.float()*2-1,torch.randn(40)*.1],1)
        pairs=[(i,j) for i in range(40) for j in range(i+1,40) if labels[i]==labels[j] and (i+j)%5==0]
        edges=torch.tensor(pairs).T.contiguous();mask=torch.arange(40)<20
        source='Synthetic transductive two-community graph'
    else:
        record=json.loads(Path(graph_file).read_text())
        labels=torch.tensor(record['labels']);features=torch.tensor(record['features'],dtype=torch.float32)
        edges=torch.tensor(record['edge_index']);mask=torch.tensor(record['train_mask'],dtype=torch.bool)
        source=record['source']
    if labels.shape!=(len(features),) or mask.shape!=labels.shape or not mask.any() or mask.all():raise ValueError('Explicit training and held-out node masks required')
    model=nil.graphs.gcn(features.shape[1],16,2)
    class Step:
        def __call__(self,network,batch):return torch.nn.functional.cross_entropy(network(features,edges)[mask],labels[mask])
    result=nil.train(model,nil.data.tensors(torch.zeros(1,1),torch.zeros(1,1)),task=Step(),epochs=100,
        batch_size=1,learning_rate=.03,device='cpu',output_dir=output/'run')
    inputs=nil.graphs.load(nil.graphs.save(features,edges,output/'graph'))
    logits=nil.predict(result,inputs,device_name='cpu')
    accuracy=float((logits[~mask].argmax(1)==labels[~mask]).float().mean())
    if accuracy<.7:raise AssertionError('Declared held-out graph quality floor failed')
    restored=nil.graphs.gcn(features.shape[1],16,2);nil.training.checkpoint.load(result.checkpoint,restored)
    torch.testing.assert_close(nil.predict(restored,inputs,device_name='cpu'),logits,rtol=0,atol=0)
    report=dict(status='passed',source=source,nodes=len(features),edges=edges.shape[1],
        training_nodes=int(mask.sum()),held_out_nodes=int((~mask).sum()),held_out_accuracy=accuracy,
        exact_reload=True,scope='Transductive homogeneous undirected node classification, not unseen-graph/heterogeneous/link prediction acceptance')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--graph-file',type=Path)
    args=parser.parse_args();print(json.dumps(exercise(args.output,graph_file=args.graph_file),indent=2))
