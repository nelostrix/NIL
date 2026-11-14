"""Token-weighted causal LM evaluation on independently prepared sequences."""

import math

from nil.runtime import device, move, require


def perplexity(model, data, batch_size=8, device_name="auto"):
    """Return next-token NLL/perplexity, excluding padding and masked labels.

    This evaluates the given windows, not sliding-window full-corpus perplexity.
    Compare results only with matching tokenizers, context and masking policy.
    """
    torch = require("torch", "torch")
    target = device(device_name)
    loader = (
        data
        if isinstance(data, torch.utils.data.DataLoader)
        else torch.utils.data.DataLoader(data, batch_size=batch_size)
    )
    previous = model.training
    model.to(target).eval()
    nll, tokens = 0.0, 0
    try:
        with torch.inference_mode():
            for raw in loader:
                if not isinstance(raw, dict) or "input_ids" not in raw or "labels" not in raw:
                    raise ValueError("Perplexity batches require input_ids and labels")
                batch = move(raw, target)
                ids, labels = batch["input_ids"], batch["labels"]
                if ids.ndim != 2 or labels.shape != ids.shape or ids.shape[1] < 2:
                    raise ValueError("Expected matching [batch, sequence>=2] IDs and labels")
                shifted = labels[:, 1:].clone()
                mask = batch.get("attention_mask")
                if mask is not None:
                    if mask.shape != ids.shape:
                        raise ValueError("attention_mask shape must match input_ids")
                    shifted[(mask[:, 1:] == 0) | (mask[:, :-1] == 0)] = -100
                valid = int((shifted != -100).sum().item())
                if not valid:
                    continue
                output = model(**{k: v for k, v in batch.items() if k != "labels"})
                logits = output["logits"] if isinstance(output, dict) else output.logits
                if logits.ndim != 3 or logits.shape[:2] != ids.shape:
                    raise ValueError("Expected [batch, sequence, vocabulary] model logits")
                value = torch.nn.functional.cross_entropy(
                    logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                    shifted.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                ).item()
                if not math.isfinite(value):
                    raise ValueError("Non-finite language-model evaluation loss")
                nll += value
                tokens += valid
    finally:
        model.train(previous)
    if not tokens:
        raise ValueError("No unmasked next-token targets to evaluate")
    average = nll / tokens
    try:
        value = math.exp(average)
    except OverflowError:
        raise ValueError("Perplexity overflow; inspect the model logits and dataset") from None
    return {"nll": average, "perplexity": value, "tokens": tokens}
