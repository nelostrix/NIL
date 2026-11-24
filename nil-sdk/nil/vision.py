"""Lazy, bounded local image classification; Pillow and Torch are explicit extras."""
from dataclasses import dataclass
from io import BytesIO
import math
import numbers
import os
from pathlib import Path

from nil.runtime import require

_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}


def _positive(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f'{name} must be an integer from 1 through {maximum}')


def _classes(values):
    if isinstance(values, (str, bytes)):
        raise ValueError("classes must be a sequence of directory names")
    values = tuple(values)
    if not 1 <= len(values) <= 1000 or any(not isinstance(v, str) or not v or v in {'.', '..'} or '/' in v or '\\' in v for v in values) or len(set(values)) != len(values):
        raise ValueError('classes must contain 1–1000 distinct directory names')
    return values


@dataclass(frozen=True)
class ImageTransform:
    """EXIF orientation, white alpha composition, RGB bilinear resize and CHW FP32.

    size is (height, width). Values start in [0,1]; optional three-channel
    normalization applies (value-mean)/std. No random augmentation or download.
    """
    size: tuple = (64, 64)
    mean: tuple | None = None
    std: tuple | None = None
    max_bytes: int = 16 * 1024 * 1024
    max_pixels: int = 4 * 1024 * 1024

    def __post_init__(self):
        if not isinstance(self.size, (tuple, list)) or len(self.size) != 2:
            raise ValueError('size must be (height, width)')
        for n in self.size:
            _positive(n, 'size dimension', 2048)
        object.__setattr__(self, 'size', tuple(self.size))
        _positive(self.max_bytes, 'max_bytes', 256 * 1024 * 1024)
        _positive(self.max_pixels, 'max_pixels', 64 * 1024 * 1024)
        if (self.mean is None) != (self.std is None):
            raise ValueError('mean and std must be supplied together')
        if self.mean is not None:
            for name in ('mean', 'std'):
                values = getattr(self, name)
                if not isinstance(values, (tuple, list)) or len(values) != 3 or any(isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v) for v in values):
                    raise ValueError(f'{name} requires three finite numbers')
                object.__setattr__(self, name, tuple(float(v) for v in values))
            if any(v <= 0 for v in self.std):
                raise ValueError('std values must be positive')

    def to_dict(self):
        return {'schema_version': 1, 'size': list(self.size),
                'mean': list(self.mean) if self.mean is not None else None,
                'std': list(self.std) if self.std is not None else None,
                'max_bytes': self.max_bytes, 'max_pixels': self.max_pixels}

    @classmethod
    def from_dict(cls, value):
        expected = {'schema_version', 'size', 'mean', 'std', 'max_bytes', 'max_pixels'}
        if not isinstance(value, dict) or set(value) != expected or type(value['schema_version']) is not int or value['schema_version'] != 1:
            raise ValueError('Unsupported image transform schema')
        return cls(**{key: v for key, v in value.items() if key != 'schema_version'})

    def __call__(self, path):
        image = require('PIL.Image', 'vision')
        image_ops = require('PIL.ImageOps', 'vision')
        torch = require('torch', 'vision')
        path = Path(path)
        # This bounds the actual read even if the file changes after discovery.
        with path.open('rb') as stream:
            encoded = stream.read(self.max_bytes + 1)
        if len(encoded) > self.max_bytes:
            raise ValueError(f'Image exceeds max_bytes: {path}')
        with image.open(BytesIO(encoded)) as source:
            if source.width * source.height > self.max_pixels:
                raise ValueError(f'Image exceeds max_pixels: {path}')
            if getattr(source, 'n_frames', 1) != 1:
                raise ValueError('Animated/multipage images require an explicit frame workflow')
            oriented = image_ops.exif_transpose(source)
            rgba = oriented.convert('RGBA')
            background = image.new('RGBA', rgba.size, (255, 255, 255, 255))
            rgb = image.alpha_composite(background, rgba).convert('RGB')
            resized = rgb.resize((self.size[1], self.size[0]), resample=image.Resampling.BILINEAR)
            pixels = bytearray(resized.tobytes())
        tensor = torch.frombuffer(pixels, dtype=torch.uint8).reshape(self.size[0], self.size[1], 3)
        tensor = tensor.permute(2, 0, 1).to(dtype=torch.float32).div_(255).contiguous()
        if self.mean is not None:
            tensor.sub_(torch.tensor(self.mean).reshape(3, 1, 1)).div_(torch.tensor(self.std).reshape(3, 1, 1))
        if not torch.isfinite(tensor).all():
            raise ValueError("Normalization produced nonfinite float32 pixels")
        return tensor


class ImageDataset:
    """Tuple-based file/label inventory with lazy decoding; no decoded image cache."""
    def __init__(self, samples, classes, transform):
        import itertools
        self.classes = _classes(classes)
        self.samples = tuple(itertools.islice(samples, 1000001))
        if not 1 <= len(self.samples) <= 1000000:
            raise ValueError('ImageDataset requires 1–1000000 samples')
        for sample in self.samples:
            if not isinstance(sample, (tuple, list)) or len(sample) != 2 or not isinstance(sample[0], (str, os.PathLike)) or type(sample[1]) is not int or not 0 <= sample[1] < len(self.classes):
                raise ValueError('Each image sample requires a local path and valid integer label')
        if not isinstance(transform, ImageTransform):
            raise TypeError('transform must be an ImageTransform')
        self.transform = transform

    @property
    def class_to_idx(self):
        return dict(zip(self.classes, range(len(self.classes))))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(path), label

    def manifest(self):
        """Persist alongside weights to retain label order and preprocessing."""
        return {'schema_version': 1, 'task': 'image_classification',
                'classes': list(self.classes), 'transform': self.transform.to_dict()}


def image_folders(root, *, classes=None, transform=None, max_samples=100000):
    """Index root/class/file without recursion or decoding. Reuse training classes.

    Unknown validation class directories fail. Known classes may be absent from
    a split. Unsupported file extensions are ignored; symlinks are rejected.
    At most max_samples+1000 directory entries are examined below class folders.
    """
    _positive(max_samples, 'max_samples', 1000000)
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('root must be an existing non-symlink directory')
    folders = {}
    with os.scandir(root) as entries:
        for index, entry in enumerate(entries):
            if index >= 1000:
                raise ValueError('Image root exceeds 1000 entries')
            if entry.is_symlink():
                raise ValueError('Image inventories must not contain symlinks')
            if entry.is_dir():
                folders[entry.name] = Path(entry.path)
    names = _classes(sorted(folders) if classes is None else classes)
    if set(folders) - set(names):
        raise ValueError('Unknown class directories; reuse the training class mapping')
    transform = ImageTransform() if transform is None else transform
    samples, examined = [], 0
    for label, name in enumerate(names):
        if name not in folders:
            continue
        with os.scandir(folders[name]) as entries:
            for entry in entries:
                examined += 1
                if examined > max_samples + 1000:
                    raise ValueError('Image directory entries exceed inventory budget')
                if entry.is_symlink():
                    raise ValueError('Image inventories must not contain symlinks')
                if entry.is_file() and Path(entry.name).suffix.lower() in _EXTENSIONS:
                    if len(samples) >= max_samples:
                        raise ValueError('Image dataset exceeds max_samples')
                    samples.append((Path(entry.path), label))
    if not samples:
        raise ValueError('No supported images found')
    samples.sort(key=lambda item: (item[1], item[0].name))
    return ImageDataset(samples, names, transform)


def classify(model, paths, *, manifest, batch_size=16, device_name='cpu'):
    """Yield per-image labels/probabilities in bounded batches using saved metadata.

    Output is an iterator; decoding and prediction occur as it is consumed.
    Inputs are local paths; the supplied model/checkpoint must match the manifest.
    """
    import itertools
    import nil
    _positive(batch_size, 'batch_size', 256)
    if not isinstance(manifest, dict) or set(manifest) != {'schema_version', 'task', 'classes', 'transform'} or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1 or manifest['task'] != 'image_classification':
        raise ValueError('Unsupported image classification manifest')
    classes = _classes(manifest['classes'])
    transform = ImageTransform.from_dict(manifest['transform'])
    torch = require('torch', 'vision')
    if isinstance(paths, (str, bytes, os.PathLike)):
        raise ValueError('paths must be an iterable of image paths, not a single path')
    iterator = iter(paths)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        logits = nil.predict(model, torch.stack([transform(path) for path in batch]), device_name=device_name)
        if not isinstance(logits, torch.Tensor) or logits.shape != (len(batch), len(classes)) or not logits.is_floating_point() or not torch.isfinite(logits).all():
            raise ValueError('Classifier must return finite [batch, classes] floating logits')
        probabilities = logits.softmax(dim=1).cpu().tolist()
        for path, scores in zip(batch, probabilities):
            index = max(range(len(classes)), key=scores.__getitem__)
            yield {'path': str(path), 'class_index': index, 'label': classes[index], 'probabilities': scores}
