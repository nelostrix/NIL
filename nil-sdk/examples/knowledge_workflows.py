"""Offline retrieval/citations, explicit tool planning and conjugate probability."""
import argparse
import json
from pathlib import Path


def exercise(output):
    import nil
    from nil.tools.base import Tool
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    index=nil.rag.Index().add('installation','NIL base installation installs only nil-sdk.',source='installation.md')
    index.add('hardware','GPU training requires an explicitly installed compatible backend.',source='hardware.md')
    artifact=index.save(output/'index');restored=nil.rag.Index.load(artifact)
    answer=restored.ask('GPU training')
    assert answer['citations']==['hardware:0']
    tool=Tool(name='power',parameters={'base':{'type':'integer'},'exp':{'type':'integer'}},fn=pow)
    plan=[dict(id='first',tool='power',arguments=dict(base=2,exp=3)),dict(id='second',tool='power',arguments=dict(base={'$ref':'first'},exp=2))]
    options=dict(allowed=['power'],journal=output/'trace.json',tool_versions={'power':'1'})
    result=nil.planning.run(plan,{'power':tool},**options)
    assert result['completed'][-1]['result']==64
    assert nil.planning.run(plan,{'power':tool},resume=True,**options)==result
    model=nil.probabilistic.BetaBernoulli().update([1,1,0,1])
    posterior=nil.probabilistic.BetaBernoulli.load(model.save(output/'posterior')).predict()
    report=dict(status='passed',scope='Small deterministic retrieval/tool/probability integration',
                answer=answer,tool_calls=len(result['completed']),tool_result=64,posterior=posterior)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(exercise(args.output),indent=2))
