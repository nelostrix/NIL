"""PyTorch-native construction helpers. All torch.nn symbols remain accessible."""

from nil.runtime import require


def __getattr__(name):
    return getattr(require("torch", "torch").nn, name)


def mlp(input_dim, output_dim, hidden=(128, 64), activation="ReLU", dropout=0.0):
    torch = require("torch", "torch")
    if any(not isinstance(n, int) or n < 1 for n in (input_dim, *hidden, output_dim)):
        raise ValueError("All layer dimensions must be positive integers")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    factory = getattr(torch.nn, activation)
    layers = []
    previous = input_dim
    for width in hidden:
        layers.extend([torch.nn.Linear(previous, width), factory()])
        if dropout:
            layers.append(torch.nn.Dropout(dropout))
        previous = width
    layers.append(torch.nn.Linear(previous, output_dim))
    return torch.nn.Sequential(*layers)


def parameters(model):
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
