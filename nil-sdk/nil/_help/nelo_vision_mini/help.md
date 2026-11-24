# Small image classifier

## Usage

```python
import nil
model = nil.model('nelo_vision_mini', num_classes=3, channels=3, width=16)
```

## Output

Two convolution stages and adaptive global average pooling; accepts float NCHW
images with spatial dimensions at least 2. Returns [batch, num_classes] logits.
Train with nil.train(..., task='classification'). No pretrained weights or
implicit downloads. Positive integer dimensions are required. This is a compact
baseline, not a claim of modern pretrained vision accuracy. See `nil help vision`
for bounded image loading and consistent preprocessing/labels at inference.

## Parameters

num_classes, channels (default3), width (default32): positive integers.

## Requirements

Explicit Torch extra; Pillow is needed only for the image-file preprocessing workflow.

## Limitations

Small randomly initialized classifier baseline; no learned backbone, detection or segmentation.
