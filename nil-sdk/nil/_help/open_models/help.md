# open_models

Optional small image-classifier builder for local acceptance experiments.

## Usage

```python
from nil.open_models.nelo_vision_mini import build

model = build(num_classes=2, channels=1, width=4)
print(type(model).__name__)
```

## Parameters

Positive class count, input channels and base width.

## Output

A native PyTorch sequential image classifier.

## Requirements

nil-sdk[torch].

## Limitations

No pretrained weights or useful accuracy guarantee; train and evaluate on appropriate data.
