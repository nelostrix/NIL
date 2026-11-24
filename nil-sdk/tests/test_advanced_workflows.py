"""Numerical references, restart and transactional failures for advanced workflows."""
import numpy as np
import pytest
import nil


class Encoder:
    fingerprint='fixture-semantic-v1'
    def encode(self,texts):
        return np.array([[int('cat' in s or 'feline' in s),int('dog' in s or 'canine' in s),.01] for s in texts],dtype=float)


def test_semantic_paraphrase_filter_restart_atomic_failure(tmp_path):
    index=nil.rag.semantic_index(Encoder(),mode='dense')
    index.add_many([dict(id='a',text='cat'),dict(id='b',text='dog')])
    assert index.search('feline',k=1)[0]['document_id']=='a'
    assert index.search('feline',allowed_documents=['b'])==[]
    path=index.save(tmp_path/'index');restored=type(index).load(path,encoder=Encoder())
    assert restored.search('canine')==index.search('canine')
    before=index.search('feline')
    index.encoder.encode=lambda texts:np.zeros((len(texts),3))
    with pytest.raises(ValueError):index.add('c','bird')
    index.encoder=Encoder()
    assert index.search('feline')==before and 'c' not in index._documents
    other=Encoder();other.fingerprint='changed'
    with pytest.raises(ValueError):type(index).load(path,encoder=other)


def test_bulk_duplicate_atomic_and_reference_answer_evaluation(tmp_path):
    index=nil.rag.Index().add('a','NIL is modular.')
    with pytest.raises(ValueError):index.add_many([dict(id='b',text='new'),dict(id='b',text='duplicate')])
    assert set(index._documents)=={'a'}
    result=nil.rag.evaluate_answers(index,[dict(question='NIL',answers=['Modular'],relevant_documents=['a'])],
        generator=lambda q,h:dict(text='modular',citations=[h[0]['id']]))
    assert result==dict(cases=1,exact_match=1.,token_f1=1.,citation_precision=1.)


def test_metropolis_moments_diagnostics_restart_and_failure(tmp_path):
    density=lambda x:-float(x@x)/2
    sampler=nil.probabilistic.metropolis(density,[[-2.],[-1.],[1.],[2.]],target_id='normal',scale=1.,seed=71)
    result=sampler.sample(4000,warmup=500)
    assert abs(result['samples'].mean())<.1 and abs(result['samples'].var()-1)<.15
    assert max(result['diagnostics']['split_rhat'])<1.05
    path=sampler.save(tmp_path/'sampler')
    restored=nil.probabilistic.load_metropolis(path,log_density=density,target_id='normal')
    np.testing.assert_array_equal(sampler.sample(20)['samples'],restored.sample(20)['samples'])
    positions=sampler.positions.copy();iterations=sampler.iterations
    sampler.log_density=lambda x:float('nan')
    with pytest.raises(ValueError):sampler.sample(2)
    np.testing.assert_array_equal(sampler.positions,positions);assert sampler.iterations==iterations
    with pytest.raises(ValueError):nil.probabilistic.metropolis(lambda x:-float('inf'),[[0.],[1.]],target_id='bad')


def test_iv_against_independent_projection_and_confounding():
    rng=np.random.default_rng(7);n=4000;z=rng.normal(size=n);u=rng.normal(size=n)
    t=1.5*z+u+rng.normal(size=n);y=2*t+3*u+rng.normal(size=n)
    result=nil.probabilistic.instrumental_ate(z,t,y)
    # Exactly identified Wald slope is an independent reference.
    expected=np.cov(z,y,ddof=0)[0,1]/np.cov(z,t,ddof=0)[0,1]
    assert result['ate']==pytest.approx(expected,abs=1e-10)
    assert abs(result['ate']-2)<.1 and result['first_stage_partial_f']>100
    assert abs(np.polyfit(t,y,1)[0]-2)>.5
    with pytest.raises(ValueError):nil.probabilistic.instrumental_ate(np.ones(n),t,y)


def test_continual_metrics_reference():
    result=nil.learning.continual_metrics([[.9,None,None],[.8,.8,None],[.6,.7,.95]])
    assert result['forgetting']==pytest.approx([.3,.1])
    assert result['mean_forgetting']==pytest.approx(.2)
    assert result['backward_transfer']==pytest.approx(-.2)
    with pytest.raises(ValueError):nil.learning.continual_metrics([[.5,.5],[.5,.5]])


def test_polyphase_attenuates_aliasing_and_pcm_roundtrip(tmp_path):
    from nil.audio import Waveform
    rate=16000;t=np.arange(rate)/rate
    wave=Waveform((.4*np.sin(2*np.pi*6000*t))[None,:].astype('float32'),rate)
    filtered=nil.audio.resample(wave,8000,method='polyphase')
    linear=nil.audio.resample(wave,8000)
    assert np.sqrt(np.mean(filtered.samples**2))<np.sqrt(np.mean(linear.samples**2))*.05
    path=tmp_path/'out.wav';nil.audio.save_wav(filtered,path)
    actual=nil.audio.load_wav(path)
    np.testing.assert_allclose(actual.samples,filtered.samples,atol=5e-5)
    assert nil.audio.word_error_rate('one two three','one four three')==pytest.approx(1/3)
    with pytest.raises(FileExistsError):nil.audio.save_wav(filtered,path)
