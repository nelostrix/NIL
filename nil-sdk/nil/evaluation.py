"""Sample-weighted metrics for common supervised tasks; custom evaluators remain native."""

import math

from nil.runtime import device, move, require


def evaluate(model, data, targets=None, task=None, batch_size=32, device_name="auto", sample_weight=None, **options):
    from nil._classical_common import is_classical, is_rl, unwrap
    if is_rl(model):
        if targets is not None or sample_weight is not None or task is not None or device_name != 'auto' or batch_size != 32:
            raise ValueError('RL evaluation expects a separate environment and episodes/max_steps options')
        from nil.rl import evaluate as evaluate_rl
        return evaluate_rl(unwrap(model), data, **options)
    if is_classical(model):
        if device_name != "auto":
            raise ValueError("Configure classical devices through native estimator parameters")
        from nil.classical import evaluate as evaluate_classical
        return evaluate_classical(model, data, targets, task=task, sample_weight=sample_weight, **options)
    neural_tasks = {"regression", "classification", "binary_classification", "multilabel_classification"}
    if isinstance(targets, str) and targets in neural_tasks and task is None:
        task, targets = targets, None
    elif isinstance(targets, str) and targets in neural_tasks and isinstance(task, int):
        # Preserve the original neural evaluate(model, data, task, batch_size, device) form.
        old_size, old_device = task, batch_size if isinstance(batch_size, str) else device_name
        task, targets, batch_size, device_name = targets, None, old_size, old_device
    if targets is not None or sample_weight is not None or options:
        raise ValueError("Neural evaluation expects a dataset/loader of (inputs, targets)")
    task = task or "regression"
    torch = require("torch", "torch")
    if task not in neural_tasks:
        raise ValueError(
            "Use regression, classification, binary_classification or multilabel_classification; provide a custom evaluator for other tasks"
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
    true_positive = false_positive = false_negative = subset_correct = sample_count = 0
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
                    if targets.is_floating_point():
                        raise ValueError("Classification metrics require integer class-index targets")
                    total += torch.nn.functional.cross_entropy(
                        output, targets, reduction="sum"
                    ).item()
                    valid = targets != -100
                    correct += ((output.argmax(dim=1) == targets) & valid).sum().item()
                    count += valid.sum().item()
                elif task in {"binary_classification", "multilabel_classification"}:
                    if output.shape != targets.shape or not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all():
                        raise ValueError("Binary/multilabel outputs must match finite 0/1 targets")
                    if task == "multilabel_classification" and targets.ndim < 2:
                        raise ValueError("Multilabel targets need a label dimension")
                    total += torch.nn.functional.binary_cross_entropy_with_logits(output, targets.to(dtype=output.dtype), reduction="sum").item()
                    predicted, actual = output >= 0, targets.bool()
                    correct += (predicted == actual).sum().item()
                    true_positive += (predicted & actual).sum().item()
                    false_positive += (predicted & ~actual).sum().item()
                    false_negative += (~predicted & actual).sum().item()
                    subset_correct += (predicted == actual).reshape(len(targets), -1).all(dim=1).sum().item()
                    sample_count += len(targets)
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
    if task in {"binary_classification", "multilabel_classification"}:
        denominator = 2 * true_positive + false_positive + false_negative
        result = {"loss": total / count, "accuracy": correct / count, "count": count,
                  "f1_micro": 2 * true_positive / denominator if denominator else 0.0}
        if task == "multilabel_classification":
            result["subset_accuracy"] = subset_correct / sample_count
        return result
    return {"mse": total / count, "rmse": math.sqrt(total / count), "count": count}
