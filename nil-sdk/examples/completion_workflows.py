"""Independent native core by default; opt in to installed neural/DDP backends."""
import argparse
import json
from pathlib import Path
import tempfile


def main():
    import nil
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neural',action='store_true'); parser.add_argument('--ddp',action='store_true')
    parser.add_argument('--fsdp',action='store_true')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='nil-completion-') as temporary:
        root=Path(temporary)
        model=nil.native.Linear(1,1,seed=7)
        result=nil.native.fit(model,[[-1.],[0.],[1.]], [[-1.],[1.],[3.]],epochs=200,lr=.15)
        path=nil.native.save(model,root/'linear.json'); restored,_=nil.native.load(path)
        assert abs(restored(nil.native.Tensor([[2.]])).item()-5)<.001
        print(json.dumps({'native_final_mse':result['history'][-1],'native_prediction':5.0}))
        if args.neural:
            learned=nil.rl.offline_fit({'observations':[[-1.],[1.]],'actions':[0,1]},action_count=2,
                hidden=(8,),steps=100,lr=.03,output_dir=root/'offline')
            bundle=nil.rl.save(learned,root/'policy')
            policy=nil.rl.load(bundle,trusted=True)
            assert nil.predict(policy,[[-1.],[1.]],deterministic=True)[0].tolist()==[0,1]
            print('Neural imitation, trusted bundle and root inference passed')
        if args.ddp or args.fsdp:
            import torch
            torch.set_num_threads(1)
            result=nil.distributed.train(torch.nn.Linear(1,1),torch.arange(8).float()[:,None]/8,
                torch.ones(8,1),epochs=2,batch_size=4,output_dir=root/'distributed',strategy='fsdp' if args.fsdp else 'ddp')
            assert len(result.history)==2
            print('Actual two-rank CPU training and committed checkpoint passed:',result.diagnostics)

if __name__=='__main__': main()
