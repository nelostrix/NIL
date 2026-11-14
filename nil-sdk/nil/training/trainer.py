"""A PyTorch-native trainer with custom steps, checkpoints and real run events."""

import math
import random
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from nil.lifecycle import Run
from nil.runtime import device, move, require, seed
from nil.training import checkpoint
from nil.training.tasks import resolve


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = "auto"
    seed: int = 0
    accumulation_steps: int = 1
    clip_grad_norm: float | None = None
    shuffle: bool = True
    output_dir: str = "./nil_runs"
    name: str = "training"

    def __post_init__(self):
        self.output_dir = str(Path(self.output_dir))
        for name in ("epochs", "batch_size", "accumulation_steps"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.clip_grad_norm is not None and (
            not math.isfinite(self.clip_grad_norm) or self.clip_grad_norm <= 0
        ):
            raise ValueError("clip_grad_norm must be finite and positive")


@dataclass
class TrainingResult:
    model: object
    run: Run
    history: list
    checkpoint: Path | None
    status: str


class Trainer:
    """Train any torch.nn.Module with a scalar-loss step(model, batch).

    epochs is the total target epoch count when resuming. Checkpoints represent
    epoch boundaries. Default map-style loaders use seed+epoch for reproducible
    ordering; external loaders/custom steps own their own non-Torch state.
    Cancellation is cooperative between batches, not process isolation.
    """

    def __init__(
        self, model, task="regression", config=None, optimizer=None, scheduler=None, callbacks=()
    ):
        self.torch = require("torch", "torch")
        self.config = config or TrainingConfig()
        self.device = device(self.config.device)
        self.model = model.to(self.device)
        self.step = resolve(task)
        self.optimizer = (
            optimizer
            if optimizer is not None
            else self.torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        )
        self.scheduler = scheduler
        self.callbacks = tuple(callbacks)
        self._cancelled = threading.Event()
        self._active = threading.Lock()
        self._finish_reason = None

    def cancel(self):
        self._cancelled.set()

    def finish(self, reason="early_stopping"):
        """Request successful completion at the next completed epoch boundary."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A nonempty completion reason is required")
        self._finish_reason = reason

    def _loader(self, data, epoch, training):
        if isinstance(data, self.torch.utils.data.DataLoader):
            return data
        generator = self.torch.Generator().manual_seed(self.config.seed + epoch)
        return self.torch.utils.data.DataLoader(
            data,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle if training else False,
            generator=generator,
        )

    def fit(self, data, validation=None, resume=None):
        if not self._active.acquire(blocking=False):
            raise RuntimeError("This Trainer is already running")
        try:
            return self._fit(data, validation, resume)
        finally:
            self._active.release()

    def _fit(self, data, validation, resume):
        self._cancelled.clear()
        self._finish_reason = None
        seed(self.config.seed)
        start, global_step, history = 0, 0, []
        if resume:
            state = checkpoint.load(
                resume, self.model, self.optimizer, self.scheduler, restore_rng=True
            )
            start, global_step = state["epoch"], state["global_step"]
            history = state.get("history", [])
            if state.get("python_rng") is not None:
                random.setstate(state["python_rng"])
            saved = state.get("config", {})
            for key in ("seed", "batch_size", "accumulation_steps", "shuffle"):
                if key in saved and saved[key] != getattr(self.config, key):
                    raise ValueError(f"Resume requires matching {key}")
        if start >= self.config.epochs:
            raise ValueError("Checkpoint already reached the configured total epochs")
        run = Run(self.config.output_dir, self.config.name, asdict(self.config))
        last_checkpoint = None
        run.status("running", device=str(self.device), resumed_from=str(resume) if resume else None)
        try:
            for epoch in range(start, self.config.epochs):
                self.model.train()
                total, batches, pending = 0.0, 0, 0
                self.optimizer.zero_grad(set_to_none=True)
                for batch in self._loader(data, epoch, True):
                    if self._cancelled.is_set():
                        self.optimizer.zero_grad(set_to_none=True)
                        run.status("cancelled", epoch=epoch, global_step=global_step)
                        return TrainingResult(
                            self.model, run, history, last_checkpoint, "cancelled"
                        )
                    loss = self.step(self.model, move(batch, self.device))
                    if not isinstance(loss, self.torch.Tensor) or loss.ndim != 0:
                        raise ValueError("Training step must return a scalar loss tensor")
                    if not self.torch.isfinite(loss).item():
                        raise ValueError(
                            "Non-finite loss; training stopped before optimizer update"
                        )
                    loss.backward()
                    total += loss.detach().item()
                    batches += 1
                    pending += 1
                    if pending == self.config.accumulation_steps:
                        self._update(pending)
                        global_step += 1
                        pending = 0
                if not batches:
                    raise ValueError("Training dataset/loader is empty")
                if pending:
                    self._update(pending)
                    global_step += 1
                metrics = {"loss": total / batches}
                if validation is not None:
                    metrics["val_loss"] = self._validate(validation)
                if self.scheduler is not None:
                    if isinstance(self.scheduler, self.torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(metrics.get("val_loss", metrics["loss"]))
                    else:
                        self.scheduler.step()
                history.append({"epoch": epoch + 1, **metrics})
                run.log(metrics, epoch + 1)
                for callback in self.callbacks:
                    callback(self, run, history[-1])
                last_checkpoint = checkpoint.save(
                    run.path / "checkpoints" / "latest.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch=epoch + 1,
                    global_step=global_step,
                    history=history,
                    config=asdict(self.config),
                    python_rng=random.getstate(),
                )
                run.event("checkpoint.saved", {"path": str(last_checkpoint)}, global_step)
                if self._cancelled.is_set():
                    run.status("cancelled", epoch=epoch + 1, global_step=global_step)
                    return TrainingResult(self.model, run, history, last_checkpoint, "cancelled")
                if self._finish_reason is not None:
                    run.status(
                        "completed",
                        epoch=epoch + 1,
                        global_step=global_step,
                        reason=self._finish_reason,
                    )
                    return TrainingResult(self.model, run, history, last_checkpoint, "completed")
            run.status("completed", epoch=self.config.epochs, global_step=global_step)
            return TrainingResult(self.model, run, history, last_checkpoint, "completed")
        except BaseException as exc:
            self.optimizer.zero_grad(set_to_none=True)
            run.status("failed", error=type(exc).__name__, message=str(exc))
            raise

    def _update(self, count):
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(count)
                if not self.torch.isfinite(parameter.grad).all().item():
                    raise ValueError("Non-finite gradient; optimizer update refused")
        if self.config.clip_grad_norm is not None:
            self.torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad_norm, error_if_nonfinite=True
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _validate(self, data):
        previous = self.model.training
        self.model.eval()
        total, count = 0.0, 0
        try:
            with self.torch.inference_mode():
                for batch in self._loader(data, 0, False):
                    value = self.step(self.model, move(batch, self.device))
                    if not self.torch.isfinite(value).all().item():
                        raise ValueError("Non-finite validation loss")
                    total += value.item()
                    count += 1
        finally:
            self.model.train(previous)
        if not count:
            raise ValueError("Validation dataset/loader is empty")
        return total / count


def train(model, data, task="regression", validation=None, resume=None, **config):
    """Convenience training; use Trainer for custom optimizers, steps and callbacks."""
    return Trainer(model, task=task, config=TrainingConfig(**config)).fit(data, validation, resume)
