# Bounded image classification

NIL now has a complete local image-classification path: file inventory, consistent
labels and preprocessing, minibatch training, held-out loss/accuracy, neural
checkpoint reload and batched per-image inference. Pillow and Torch are explicit
`vision` extras; the base NIL package keeps zero dependencies. No pretrained
weights, images, background decoder or network service starts on import.

```sh
python -m pip install '.[vision]'
python nil-sdk/examples/image_classification.py \
  --train /data/images/train --valid /data/images/valid --output /tmp/image-run
nil help vision
nil workflows vision.classification --json
```

Place files at `root/class_name/file.png` (JPEG and other Pillow codecs are also
supported). The training inventory sorts classes; validation uses that same order.
Unknown validation classes fail. Missing known classes retain their original
indices. Indexing stores paths and labels, decodes lazily and ignores unsupported
extensions. Nested directories are not scanned. Symlinks at discovery are rejected;
this trusted-local-file workflow is not a hostile decoder sandbox.

`ImageTransform` applies EXIF orientation, converts to RGB with transparency over
white, resizes bilinearly to `(height,width)`, and returns contiguous float32 CHW
pixels in [0,1]. Optional three-channel mean/std normalization is explicit and
stored in the manifest. There is no stochastic augmentation. PNG/JPEG, grayscale,
RGBA/palette and EXIF tests fix these semantics. Animated/multipage, malformed,
nonfinite preprocessing and over-budget inputs fail. Codec support follows the
installed Pillow build.

Default per-file limits: 16MiB encoded read and 4,194,304 source pixels. Output
dimensions are at most 2048 per side. Inventory limits default to 100,000 samples,
1000 root entries and at most max_samples+1000 examined class-folder entries.
These are input/work budgets, not hard whole-process RAM limits. Model, batch,
decoded intermediates and backend runtime also use memory. Use small batches and
zero loader workers on constrained machines. `classify` yields bounded batches,
so callers can consume long path iterators without retaining all predictions.

`dataset.manifest()` contains schema version, task, class order and transform
configuration. Save it beside the neural checkpoint; inference must use the same
architecture, class mapping and transform. `ImageTransform.from_dict` rejects
unknown schema/config keys. The example reloads both JSON and weights and checks
identical probabilities. A matching class count alone cannot detect swapped model
weights; provenance and complete signed artifact packaging remain broader work.

## Validation and limits

`test_vision.py` covers pixel values, EXIF, corrupt/multipage files, byte/pixel
limits, lazy indexing and file changes, split label mismatch, bounded inference,
actual learning, evaluation and exact checkpoint reload. The no-argument example
uses eight color images to check wiring; it is not a held-out quality result.

For a real offline dataset, explicitly prepare sklearn's bundled handwritten
digits in an environment where sklearn and Pillow have been installed:

```sh
python scripts/accept_vision.py prepare-digits --output /tmp/nil-digits
# In a separate environment with the regular NIL wheel and vision extras:
python -I scripts/accept_vision.py check --dataset /tmp/nil-digits \
  --output /tmp/nil-digits-acceptance
```

This fixed split uses 1000 training and 300 held-out real glyphs (seed41), grayscale
PNG conversion, 60 epochs and a predeclared minimum 80% held-out accuracy. The
acceptance report records versions, source indices, dataset fingerprint, wall time,
child peak RSS and exact checkpoint reload. It does not establish natural-image
accuracy, older-PC suitability, accelerator performance or a hard RAM quota.

Detection, segmentation, tracking, OCR, image retrieval, learned/pretrained visual
encoders, augmentation, export and serving acceptance remain unchecked in the
[32-area roadmap](project/TODO.md). Image/video generation belongs to generative
workflows and is not implemented by this classification pipeline.

Recorded [2026-11-24 acceptance](benchmarks/2026-11-24-vision-digits.json):
89.33% held-out accuracy, loss 2.31073 → 0.37937, exact prediction reload,
29.93 seconds and 333,979,648-byte peak child RSS on Linux/Python3.12.16,
Torch2.15.0+cpu, NumPy2.5.4 and Pillow13.0.0. This is a measured current-host
result without a hard memory/CPU quota.
