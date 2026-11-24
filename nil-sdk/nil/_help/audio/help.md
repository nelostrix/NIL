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

Uncompressed8/16/24/32-bit PCM only. Default32MiB/60seconds/8channels. Linear resampling is not bandlimited; no ASR/TTS model or compressed-audio decoder is implied.

See docs/SEVEN_WORKFLOWS.md and the bundled workflow examples for complete protocols and acceptance.
