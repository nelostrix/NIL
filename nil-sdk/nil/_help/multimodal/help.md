# multimodal

## Usage

```python
import nil
clip = nil.multimodal.load_clip('animation.gif', size=(32,32))
visual = nil.multimodal.visual_features(clip)
# video = nil.multimodal.load_video('clip.mp4', job_root='video_jobs')
fused = nil.multimodal.fuse_features({'visual': visual}, {'visual':9})
```

## Parameters

Frame/byte/pixel/decoded budgets, image size and durations; load_video fps/timeout/job_root; explicit feature dimensions and missing policy.

## Output

TCHW Clip/timing; visual summary; aligned audio segments; fixed-order fused vector plus presence mask.

## Requirements

GIF/TIFF needs vision extra; fusion/audio alignment needs NumPy; MP4/WebM additionally needs explicitly installed FFmpeg/ffprobe.

## Limitations

MP4 defaults to constant-rate sampling; original timestamps require preserve_timestamps=True and available source PTS/duration. Local CLIP requires explicit weights/Transformers. Language generation, semantic video training and broad quality acceptance remain open. Default decoded cap64MiB; source is trusted local data.

See docs/SEVEN_WORKFLOWS.md for v1 acceptance and docs/ADVANCED_WORKFLOWS.md for this phase and remaining work.

## Local CLIP and original timestamps

load_encoder(local_dir) loads a native CPU CLIP with explicit Transformers; encode_text and encode_images(PIL_images) return normalized joint-space vectors. load_video(..., preserve_timestamps=True) requires original PTS and last-frame duration, or refuses. Default remains CFR; video-language training remains open.
