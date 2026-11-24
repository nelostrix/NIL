"""Functional MAML adaptation; caller-owned outer optimizer and task sampling."""
from nil.runtime import require


def adapt(model, inputs, targets, loss, *, steps=1, lr=.01, first_order=False):
    """Return adapted parameters without mutating the model or its gradients.

    Deterministic eval-mode, buffer-free modules only. Pass returned parameters to
    torch.func.functional_call. Second-order gradients remain connected by default.
    """
    torch=require('torch','torch')
    if type(steps) is not int or not 1<=steps<=20 or type(lr) not in (int,float) or not 0<lr<=10 or type(first_order) is not bool or not callable(loss):raise ValueError('Invalid adaptation configuration')
    if model.training or any(True for _ in model.buffers()):raise ValueError('Use an eval-mode buffer-free model for functional adaptation')
    parameters=dict(model.named_parameters())
    if not parameters or sum(p.numel() for p in parameters.values())>1000000 or not all(p.requires_grad for p in parameters.values()):raise ValueError('Require 1..1M trainable parameters')
    for _ in range(steps):
        prediction=torch.func.functional_call(model,parameters,(inputs,))
        value=loss(prediction,targets)
        if value.ndim or not torch.isfinite(value):raise ValueError('Loss must be a finite scalar')
        gradients=torch.autograd.grad(value,tuple(parameters.values()),create_graph=not first_order)
        if any(not torch.isfinite(g).all() for g in gradients):raise ValueError('Nonfinite support gradient')
        parameters={name:p-lr*g for (name,p),g in zip(parameters.items(),gradients)}
    return parameters
