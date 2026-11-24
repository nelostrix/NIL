# audio

## Usage

```python
import nil
sound = nil.audio.load_wav('recording.wav')
print(sound.sample_rate, sound.samples.shape)
features = nil.audio.features(sound)
```

## Parameters

Actual read/seconds limits; PCM sample rate/channels; explicit resample target; segment boundaries; spectral bands/n_fft/hop.

## Output

Float32 channel-first Waveform, bounded waveform segment/resample, fixed-length mean/std log-power/RMS/ZCR feature vector.

## Requirements

Explicit audio extra (NumPy); WAV parser is standard library.

## Limitations

WAV decoding supports uncompressed8/16/24/32-bit PCM, default32MiB/60seconds/8channels. Default linear resampling is not bandlimited; polyphase is an explicit SciPy option. ASR/TTS require separately supplied local models and backends; compressed-audio decoding and streaming remain open.

See docs/SEVEN_WORKFLOWS.md for v1 acceptance and docs/ADVANCED_WORKFLOWS.md for this phase and remaining work.

## Local speech and anti-alias resampling

load_asr(local_dir) loads native CTC or explicit seq2seq; mono input, matching processor rate and at most 30 seconds per call. load_tts(local_dir).synthesize(text, seed=42) uses VITS. Explicit Transformers/Torch required. resample(..., method="polyphase") requires SciPy; normalize headroom explicitly if interpolation overshoots. save_wav creates a new PCM16 file. word_error_rate compares normalized reference/hypothesis words; accuracy is not guaranteed.

## Incremental speech recognition

```python
recognizer = nil.audio.load_asr('/models/local-ctc')
stream = nil.audio.stream(recognizer, window_seconds=5., min_tail_seconds=.1)
# records = stream.push(mono_waveform_at_the_model_rate)
# state = stream.save('/artifacts/new-speech-state')
# stream = nil.audio.load_stream(state, recognizer=recognizer)
# final_records = stream.flush()
```

Pushes may end at arbitrary samples. Complete fixed windows return text plus absolute sample/time intervals; pending samples stay bounded below one window. Flush closes the stream and refuses too-short final tails. Save/load binds model identity, configuration, pending audio and offsets. Decoder failure leaves stream state unchanged. This is fixed-window inference, not a recurrent ASR decoder: boundary words can be lost, and VAD, diarization, microphone capture and downstream exactly-once delivery remain open. See docs/CONTINUED_WORKFLOWS.md for all budgets and recovery semantics.
