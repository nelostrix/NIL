"""Task adapters. Supply a custom step for arbitrary architectures or objectives."""

from nil.runtime import require


class Supervised:
    def __init__(self, loss="mse"):
        torch = require("torch", "torch")
        choices = {
            "mse": torch.nn.MSELoss,
            "cross_entropy": torch.nn.CrossEntropyLoss,
            "bce_logits": torch.nn.BCEWithLogitsLoss,
            "l1": torch.nn.L1Loss,
        }
        self.loss = choices[loss]() if isinstance(loss, str) else loss

    def __call__(self, model, batch):
        if not isinstance(batch, (list, tuple)) or len(batch) != 2:
            raise ValueError(
                "Supervised batches must be (inputs, targets); use a custom step otherwise"
            )
        inputs, targets = batch
        if isinstance(inputs, dict):
            prediction = model(**inputs)
        elif isinstance(inputs, (list, tuple)):
            prediction = model(*inputs)
        else:
            prediction = model(inputs)
        torch = require("torch", "torch")
        if isinstance(self.loss, (torch.nn.MSELoss, torch.nn.L1Loss, torch.nn.BCEWithLogitsLoss)):
            if prediction.shape != targets.shape:
                raise ValueError(
                    "Output and target shapes must match; implicit broadcasting is disabled"
                )
        return self.loss(prediction, targets)


class CausalLM:
    """Hugging Face-compatible causal LM batch with explicit labels.

    Use nil.generative.text_dataset to mask padding correctly. No assumption that
    padding token IDs differ from EOS: an attention mask carries that distinction.
    """

    def __call__(self, model, batch):
        if not isinstance(batch, dict) or "labels" not in batch:
            raise ValueError("CausalLM batches require input_ids and labels (padding labels=-100)")
        output = model(**batch)
        loss = output.get("loss") if isinstance(output, dict) else output.loss
        if loss is None:
            raise ValueError("Causal model did not return a loss")
        return loss


def resolve(task):
    if task in ("regression", "imitation_learning"):
        return Supervised("mse")
    if task == "classification":
        return Supervised("cross_entropy")
    if task == "causal_lm":
        return CausalLM()
    if callable(task):
        return task
    raise ValueError(
        "Choose regression, classification, imitation_learning, causal_lm or a callable"
    )
