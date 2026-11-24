"""Block-invariant fixed-window recognition and durable sample boundaries."""
import numpy as np
import pytest
import nil


class Recognizer:
    sample_rate=100
    fingerprint='deterministic-audio-fixture-v1'
    def transcribe(self,wave):
        return dict(text=str(round(float(wave.samples.sum()),5)))


def waveform(values):return nil.audio.Waveform(np.asarray(values,dtype=np.float32).reshape(1,-1),100)


def test_arbitrary_block_boundaries_restart_and_no_duplicate_samples(tmp_path):
    samples=np.arange(73,dtype=np.float32)/100
    full=nil.audio.stream(Recognizer(),window_seconds=.2,min_tail_seconds=0)
    expected=full.push(waveform(samples))+full.flush()
    partial=nil.audio.stream(Recognizer(),window_seconds=.2,min_tail_seconds=0)
    actual=partial.push(waveform(samples[:13]));path=partial.save(tmp_path/'stream')
    resumed=nil.audio.load_stream(path,recognizer=Recognizer())
    for first,last in [(13,23),(23,57),(57,73)]:actual+=resumed.push(waveform(samples[first:last]))
    actual+=resumed.flush()
    assert actual==expected
    assert [(r['start_sample'],r['end_sample']) for r in actual]==[(0,20),(20,40),(40,60),(60,73)]
    closed=resumed.save(tmp_path/'closed');restored=nil.audio.load_stream(closed,recognizer=Recognizer())
    with pytest.raises(ValueError,match='closed'):restored.push(waveform([0.]))
    other=Recognizer();other.fingerprint='different'
    with pytest.raises(ValueError,match='mismatch'):nil.audio.load_stream(path,recognizer=other)


def test_failed_decode_preserves_pending_and_counter():
    recognizer=Recognizer();stream=nil.audio.stream(recognizer,window_seconds=.2,min_tail_seconds=0)
    stream.push(waveform([.1]*10))
    def failed(wave):raise RuntimeError('decoder failure')
    recognizer.transcribe=failed
    with pytest.raises(RuntimeError):stream.push(waveform([.1]*30))
    assert stream.offset==0 and stream._pending.shape==(1,10)
    recognizer.transcribe=Recognizer().transcribe
    rows=stream.push(waveform([.1]*30));assert len(rows)==2 and stream.offset==40


def test_tail_and_push_limits_are_explicit_and_transactional():
    stream=nil.audio.stream(Recognizer(),window_seconds=.2,min_tail_seconds=.1,max_windows_per_push=1)
    with pytest.raises(ValueError,match='budget'):stream.push(waveform([0.]*40))
    assert stream.offset==0
    stream.push(waveform([0.]*5))
    with pytest.raises(ValueError,match='minimum'):stream.flush()
    assert not stream.closed
    stream.push(waveform([0.]*5));assert stream.flush()[0]['end_sample']==10
    recognizer=Recognizer();recognizer.sample_rate=1
    with pytest.raises(ValueError,match='one sample'):nil.audio.stream(recognizer,window_seconds=.1,min_tail_seconds=0)
