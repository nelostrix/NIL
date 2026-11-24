"""Bounded offline statistical/semantic workflows; explicit NumPy backend only."""
import argparse
import json
from pathlib import Path
import numpy as np
import nil


class VocabularyEncoder:
    """Illustrative caller encoder, not a pretrained semantic-quality claim."""
    fingerprint='example-vocabulary-1'
    def encode(self,texts):
        return np.asarray([[int('cat' in s or 'feline' in s),int('planet' in s or 'space' in s),.01] for s in texts],dtype=float)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    encoder=VocabularyEncoder();index=nil.rag.semantic_index(encoder,mode='dense')
    index.add_many([dict(id='pet',text='cat indoors'),dict(id='sky',text='planet in space')])
    saved=index.save(args.output/'index');restored=type(index).load(saved,encoder=encoder)
    assert restored.search('feline',k=1)[0]['document_id']=='pet'
    density=lambda x:-float(x@x)/2
    sampler=nil.probabilistic.metropolis(density,[[-1.],[1.]],target_id='standard-normal',seed=7,scale=1.)
    draws=sampler.sample(1500,warmup=200);sampler.save(args.output/'chains')
    rng=np.random.default_rng(7);z=rng.normal(size=2000);u=rng.normal(size=2000)
    treatment=2*z+u+rng.normal(size=2000);outcome=3*treatment+u+rng.normal(size=2000)
    iv=nil.probabilistic.instrumental_ate(z,treatment,outcome)
    assert abs(iv['ate']-3)<.15 and abs(draws['samples'].mean())<.2
    report=dict(semantic_hit='pet',sampler_mean=float(draws['samples'].mean()),sampler_diagnostics=draws['diagnostics'],
                instrumental_effect=iv,continual=nil.learning.continual_metrics([[.9,None],[.8,.85]]))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
