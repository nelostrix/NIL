"""Small trainable image classifier; no pretrained weights or implicit downloads."""
from nil.runtime import require


def build(num_classes, channels=3, width=32):
    torch = require("torch", "torch")
    if any(type(n) is not int or n < 1 for n in (num_classes, channels, width)):
        raise ValueError("Class, channel and width dimensions must be positive")
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 2, num_classes),
    )
