"""Sample-weighted metrics for common supervised tasks; custom evaluators remain native."""

import math

from nil.runtime import device, move, require


def evaluate(model, data, task="regression", batch_size=32, device_name="auto"):
    torch = require("torch", "torch")
    if task not in {"regression", "classification"}:
        raise ValueError(
            "Use regression or classification; provide a custom evaluator for other tasks"
        )
    target = device(device_name)
    loader = (
        data
        if isinstance(data, torch.utils.data.DataLoader)
        else torch.utils.data.DataLoader(data, batch_size=batch_size)
    )
    previous = model.training
    model.to(target).eval()
    total, count, correct = 0.0, 0, 0
    try:
        with torch.inference_mode():
            for inputs, targets in loader:
                inputs, targets = move(inputs, target), move(targets, target)
                if isinstance(inputs, dict):
                    output = model(**inputs)
                elif isinstance(inputs, (tuple, list)):
                    output = model(*inputs)
                else:
                    output = model(inputs)
                if task == "classification":
                    total += torch.nn.functional.cross_entropy(
                        output, targets, reduction="sum"
                    ).item()
                    correct += (output.argmax(dim=1) == targets).sum().item()
                    count += targets.numel()
                else:
                    if output.shape != targets.shape:
                        raise ValueError("Regression output and target shapes must match")
                    total += torch.nn.functional.mse_loss(output, targets, reduction="sum").item()
                    count += targets.numel()
    finally:
        model.train(previous)
    if not count:
        raise ValueError("Evaluation dataset/loader is empty")
    if not math.isfinite(total):
        raise ValueError("Non-finite evaluation result")
    if task == "classification":
        return {"loss": total / count, "accuracy": correct / count, "count": count}
    return {"mse": total / count, "rmse": math.sqrt(total / count), "count": count}
