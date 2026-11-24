# Image classification

## Requirements

Opt in with `pip install 'nil-sdk[vision]'` in the environment where you want
Pillow and Torch. Installing NIL alone still installs NIL only.

## Usage

```python
import nil
transform = nil.vision.ImageTransform(size=(64, 64))  # height, width
train = nil.vision.image_folders('images/train', transform=transform)
valid = nil.vision.image_folders('images/valid', classes=train.classes, transform=transform)
model = nil.model('nelo_vision_mini', num_classes=len(train.classes), width=16)
result = nil.train(model, train, task='classification', validation=valid,
                   epochs=10, batch_size=16, device='cpu', output_dir='image_runs')
print(nil.evaluate(result, valid, task='classification', device_name='cpu'))
manifest = train.manifest()  # save as JSON beside result.checkpoint
for prediction in nil.vision.classify(result, ['example.jpg'], manifest=manifest):
    print(prediction['label'], prediction['probabilities'])
```

## Parameters

Folders are root/class/file; indexing sorts labels and filenames without decoding.
Reuse training classes for validation/test; unknown classes fail, absent known
classes keep their training indices. Decoding happens per item, not all at once.
PNG/JPEG/BMP/WebP/single-page TIFF depend on installed Pillow codec support.
Malformed, animated/multipage and over-budget images fail on access.

## Limitations

Transform semantics: apply EXIF orientation; convert grayscale/palette/RGBA to
RGB, composite transparency over white, bilinear resize, contiguous float32 CHW
pixels in [0,1]. Optional three-channel mean/std applies (pixels-mean)/std. No
random augmentation. ImageTransform.to_dict/from_dict roundtrips schema-1 JSON.
Default limits are 16MiB encoded bytes and 4,194,304 source pixels per file,
100,000 samples per inventory, 1000 root entries and bounded folder entry scanning.
These bound individual reads and inventory size, not total process RAM. Use small
batches and zero workers on constrained PCs. No decoded cache or background worker
starts on import. Input files are trusted local files and may change after scan;
the actual read remains byte-bounded. This is not a hostile-file sandbox.

## Output

classify returns an iterator with bounded batches; invalid output shape or
nonfinite logits fail. Manifest labels must match model weights. Persist the
manifest with the neural checkpoint; restore weights into the same architecture.
Detection, segmentation, tracking, OCR, retrieval, pretrained vision backbones,
export and serving acceptance remain separate roadmap work. See docs/VISION.md.
