"""Continual replay/recovery, heterogeneous graphs and incremental speech.

All default data is synthetic. --asr-model and --audio explicitly opt into an
existing local CTC model and recorded WAV; no downloads or installs occur.
"""
import argparse
import json
from pathlib import Path
import time


def task_data(seed):
    import torch
    generator=torch.Generator().manual_seed(seed);tasks=[]
    for stage in range(2):
        train=torch.randn(64,2,generator=generator);test=torch.randn(64,2,generator=generator)
        train_labels=(train[:,stage]>0).long();test_labels=(test[:,stage]>0).long()
        train=torch.cat([train,torch.full((64,1),float(2*stage-1))],1)
        test=torch.cat([test,torch.full((64,1),float(2*stage-1))],1)
        tasks.append(dict(id=f'context-{stage}',train=(train,train_labels),test=(test,test_labels)))
    return tasks


def exercise(output,asr_model=None,audio=None):
    import torch,numpy as np,nil
    from importlib.metadata import version
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);started=time.monotonic()
    report=dict(schema_version=1,scope='Synthetic CPU training and functional stream recovery; optional real CTC execution is not a transcript-quality benchmark',versions={n:version(n) for n in ('torch','numpy','nil-sdk')},continual=[])
    def model(seed):
        torch.manual_seed(seed)
        return torch.nn.Sequential(torch.nn.Linear(3,12),torch.nn.Tanh(),torch.nn.Linear(12,2))
    for seed in (7,11,19):
        data=task_data(seed)
        for capacity in (0,32):
            result=nil.learning.continual_fit(model(seed),data,classes=2,epochs=10,batch_size=16,learning_rate=.03,replay_capacity=capacity,seed=seed,output_dir=output/f'continual-{seed}-{capacity}')
            report['continual'].append(dict(seed=seed,replay_capacity=capacity,accuracy=result['accuracy'],metrics=result['metrics'],seen=result['replay']['seen'],optimizer_updates=80 if capacity==0 else 100,example_exposures=1280 if capacity==0 else 1600,budget_note='Same epochs, additional replay examples: not compute-matched'))
    # Actual training on relation-only signal; held-out paper labels never fit.
    torch.manual_seed(17);relation=('author','writes','paper')
    features={'author':torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]]),'paper':torch.zeros(8,1)}
    edges={relation:torch.tensor([[0,1,2,3,0,1,2,3],list(range(8))])};labels=torch.arange(8)%2
    graph_model=nil.graphs.hetero_sage({'author':2,'paper':1},8,{'author':2,'paper':2},[relation])
    optimizer=torch.optim.Adam(graph_model.parameters(),lr=.03)
    for _ in range(40):
        optimizer.zero_grad();logits=graph_model(features,edges)['paper']
        loss=torch.nn.functional.cross_entropy(logits[:4],labels[:4]);loss.backward();optimizer.step()
    with torch.no_grad():expected=graph_model(features,edges)['paper'];accuracy=float((expected[4:].argmax(1)==labels[4:]).float().mean())
    assert accuracy==1.
    graph_path=nil.graphs.save_hetero(features,edges,output/'typed-graph');nil.training.checkpoint.save(output/'typed-model.pt',graph_model)
    restored=nil.graphs.hetero_sage({'author':2,'paper':1},8,{'author':2,'paper':2},[relation]);nil.training.checkpoint.load(output/'typed-model.pt',restored)
    torch.testing.assert_close(expected,restored(**nil.graphs.load_hetero(graph_path))['paper'],rtol=0,atol=0)
    report['heterogeneous_graph']=dict(heldout_accuracy=accuracy,heldout_nodes=4,scope='Synthetic typed relational classification, transductive topology',exact_reload=True)
    if asr_model:
        recognizer=nil.audio.load_asr(asr_model);wave=nil.audio.load_wav(audio)
        wave=nil.audio.Waveform(wave.samples*.8,wave.sample_rate)
        wave=nil.audio.resample(nil.audio.mono(wave),recognizer.sample_rate)
    else:
        class Recognizer:
            fingerprint='example-amplitude-sum-v1';sample_rate=100
            def transcribe(self,wave):return dict(text=str(round(float(wave.samples.sum()),5)))
        recognizer=Recognizer();wave=nil.audio.Waveform(np.linspace(0,.5,340,dtype=np.float32)[None,:],100)
    direct=nil.audio.stream(recognizer,window_seconds=1.,min_tail_seconds=.1)
    expected=direct.push(wave)+direct.flush()
    stream=nil.audio.stream(recognizer,window_seconds=1.,min_tail_seconds=.1)
    cut=min(wave.samples.shape[1]-1,max(1,int(recognizer.sample_rate*.37)))
    actual=stream.push(nil.audio.Waveform(wave.samples[:,:cut],wave.sample_rate));saved=stream.save(output/'speech-state')
    stream=nil.audio.load_stream(saved,recognizer=recognizer)
    actual+=stream.push(nil.audio.Waveform(wave.samples[:,cut:],wave.sample_rate))+stream.flush()
    assert actual==expected and actual[-1]['end_sample']==wave.samples.shape[1]
    report['speech']=dict(real_model=bool(asr_model),records=actual,exact_block_invariance_and_reload=True,wer=None)
    report['seconds']=time.monotonic()-started
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--asr-model',type=Path);parser.add_argument('--audio',type=Path)
    args=parser.parse_args()
    if bool(args.asr_model)!=bool(args.audio):parser.error('Provide --asr-model and --audio together')
    print(json.dumps(exercise(args.output,args.asr_model,args.audio),indent=2))


if __name__=='__main__':main()
