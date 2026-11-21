"""A PyTorch-native trainer with custom steps, checkpoints and real run events."""

import math
import random
import threading
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

from nil.lifecycle import Run
from nil.runtime import device, move, require, seed
from nil.training import checkpoint
from nil.training.tasks import resolve
from nil.training._loaders import LoaderContract


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
    precision: str = "fp32"
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = False
    persistent_workers: bool = False
    max_batches_per_epoch: int | None = None

    def __post_init__(self):
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16 or fp16")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or not 0 <= self.num_workers <= 8:
            raise ValueError('num_workers must be between zero and eight')
        if isinstance(self.prefetch_factor, bool) or not isinstance(self.prefetch_factor, int) or not 1 <= self.prefetch_factor <= 8:
            raise ValueError('prefetch_factor must be between one and eight')
        if not isinstance(self.pin_memory, bool) or not isinstance(self.persistent_workers, bool):
            raise ValueError('pin_memory and persistent_workers must be booleans')
        if self.persistent_workers:
            raise ValueError('Persistent workers are unsupported for exact RNG resume')
        if self.max_batches_per_epoch is not None and (isinstance(self.max_batches_per_epoch, bool) or not isinstance(self.max_batches_per_epoch, int) or self.max_batches_per_epoch < 1):
            raise ValueError('max_batches_per_epoch must be a positive integer')
        self.output_dir = str(Path(self.output_dir))
        for name in ("epochs", "batch_size", "accumulation_steps"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.learning_rate, bool) or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
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
        if not isinstance(model, self.torch.nn.Module):
            raise TypeError("nil.train requires a torch.nn.Module; use nil.fit for classical models or nil.rl.train for control algorithms")
        self.config = config or TrainingConfig()
        self.device = device(self.config.device)
        if self.config.precision != "fp32" and self.device.type not in {"cpu", "cuda"}:
            raise NotImplementedError("Mixed precision currently supports CPU/CUDA only")
        if self.config.precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 training requires CUDA; use bf16 or fp32 on CPU")
        if self.config.precision == "bf16" and self.device.type == "cuda" and not self.torch.cuda.is_bf16_supported():
            raise ValueError("CUDA bf16 is unavailable on this device")
        self.model = model.to(self.device)
        self.step = resolve(task)
        if isinstance(getattr(self.step, "loss", None), self.torch.nn.Module):
            self.step.loss.to(self.device)
        self.optimizer = (
            optimizer
            if optimizer is not None
            else self.torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        )
        self.scheduler = scheduler
        members = {id(parameter) for parameter in self.model.parameters()}
        optimized = [parameter for group in self.optimizer.param_groups for parameter in group['params']]
        if not optimized or len({id(parameter) for parameter in optimized}) != len(optimized) or any(id(parameter) not in members for parameter in optimized):
            raise ValueError('Optimizer parameters must be unique members of this model; construct custom optimizers after moving the model to its device')
        if scheduler is not None and getattr(scheduler, 'optimizer', self.optimizer) is not self.optimizer:
            raise ValueError('Scheduler must be bound to this trainer optimizer')
        self.scaler = self.torch.amp.GradScaler("cuda", enabled=self.config.precision == "fp16")
        self._amp_skipped = 0
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
        contract = self._data_contract if training else self._validation_contract
        return contract.loader(epoch)

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
        self._data_contract = LoaderContract(data, self.config, self.torch)
        self._validation_contract = LoaderContract(validation, self.config, self.torch, False) if validation is not None else None
        start, global_step, history = 0, 0, []
        if resume:
            payload = checkpoint.read(resume)
            state = payload["state"]
            start, global_step = state["epoch"], state["global_step"]
            history = state.get("history", [])
            saved = state.get("config", {})
            for key in ("seed", "batch_size", "accumulation_steps", "shuffle", "learning_rate", "clip_grad_norm"):
                if key in saved and saved[key] != getattr(self.config, key):
                    raise ValueError(f"Resume requires matching {key}")
            if saved.get("precision", "fp32") != self.config.precision:
                raise ValueError("Resume requires matching precision")
            loader_states = state.get('loader_states')
            if loader_states is None:
                # Legacy checkpoints only captured the owned map-loader RNG contract.
                if self._data_contract.external or self._data_contract.stateful or self.config.num_workers or self.config.max_batches_per_epoch is not None:
                    raise ValueError('Checkpoint does not contain the required loader state')
            else:
                for name, contract in [('train', self._data_contract), ('validation', self._validation_contract)]:
                    saved_loader = loader_states.get(name)
                    if (contract is None) != (saved_loader is None):
                        raise ValueError('Resume requires matching validation data contract')
                    if contract is not None:
                        contract.validate_state_dict(saved_loader)
            if self.scaler.is_enabled() and not state.get("amp_scaler"):
                raise ValueError("AMP resume requires saved GradScaler state")
            if self.scaler.is_enabled():
                restored_scaler = self.torch.amp.GradScaler("cuda")
                restored_scaler.load_state_dict(state["amp_scaler"])
            if state.get("python_rng") is not None:
                random.Random().setstate(state["python_rng"])
        if start >= self.config.epochs:
            raise ValueError("Checkpoint already reached the configured total epochs")
        if resume:
            checkpoint.restore(payload, self.model, self.optimizer, self.scheduler, restore_rng=True)
            if state.get('loader_states') is not None:
                self._data_contract.load_state_dict(state['loader_states']['train'])
                if self._validation_contract is not None:
                    self._validation_contract.load_state_dict(state['loader_states']['validation'])
            if self.scaler.is_enabled():
                self.scaler = restored_scaler
            self._amp_skipped = state.get("amp_skipped_updates", 0)
            if state.get("python_rng") is not None:
                random.setstate(state["python_rng"])
        else:
            seed(self.config.seed)
            self._amp_skipped = 0
        run = Run(self.config.output_dir, self.config.name, asdict(self.config))
        last_checkpoint = None
        run.status("running", device=str(self.device), resumed_from=str(resume) if resume else None)
        try:
            for epoch in range(start, self.config.epochs):
                skipped_before = self._amp_skipped
                self.model.train()
                total, denominator, batches, pending, pending_weight = 0.0, 0.0, 0, 0, 0.0
                self.model.zero_grad(set_to_none=True)
                for batch in self._loader(data, epoch, True):
                    if self._cancelled.is_set():
                        self.model.zero_grad(set_to_none=True)
                        run.status("cancelled", epoch=epoch, global_step=global_step)
                        return TrainingResult(
                            self.model, run, history, last_checkpoint, "cancelled"
                        )
                    batch = move(batch, self.device)
                    with self._autocast():
                        loss = self.step(self.model, batch)
                    if not isinstance(loss, self.torch.Tensor) or loss.ndim != 0:
                        raise ValueError("Training step must return a scalar loss tensor")
                    if not self.torch.isfinite(loss).item():
                        raise ValueError(
                            "Non-finite loss; training stopped before optimizer update"
                        )
                    weight = self._weight(batch)
                    weighted_loss = loss * weight
                    if self.scaler.is_enabled():
                        self.scaler.scale(weighted_loss).backward()
                    else:
                        weighted_loss.backward()
                    total += loss.detach().item() * weight
                    denominator += weight
                    batches += 1
                    pending += 1
                    pending_weight += weight
                    if pending == self.config.accumulation_steps:
                        global_step += int(self._update(pending_weight))
                        pending = 0
                        pending_weight = 0.0
                if not batches:
                    raise ValueError("Training dataset/loader is empty")
                if pending:
                    global_step += int(self._update(pending_weight))
                metrics = {"loss": total if getattr(self.step, "reduction", "mean") == "sum" else total / denominator}
                if self._amp_skipped > skipped_before:
                    metrics["amp_skipped_updates"] = self._amp_skipped - skipped_before
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
                    amp_scaler=self.scaler.state_dict(),
                    amp_skipped_updates=self._amp_skipped,
                    loader_states={'train': self._data_contract.state_dict(),
                                   'validation': self._validation_contract.state_dict() if self._validation_contract else None},
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
            self.model.zero_grad(set_to_none=True)
            run.status("failed", error=type(exc).__name__, message=str(exc))
            raise

    def _update(self, count):
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        if getattr(self.step, "reduction", "mean") == "sum":
            count = 1.0
        optimized = [parameter for group in self.optimizer.param_groups for parameter in group['params']]
        for parameter in optimized:
            if parameter.grad is not None:
                parameter.grad.div_(count)
                if not self.torch.isfinite(parameter.grad).all().item():
                    if self.scaler.is_enabled():
                        # Native GradScaler skips the unsafe optimizer step and
                        # reduces its scale. Do not count it as a completed update.
                        self.scaler.update(new_scale=self.scaler.get_scale() * self.scaler.get_backoff_factor())
                        self._amp_skipped += 1
                        self.model.zero_grad(set_to_none=True)
                        return False
                    raise ValueError("Non-finite gradient; optimizer update refused")
        if self.config.clip_grad_norm is not None:
            self.torch.nn.utils.clip_grad_norm_(
                optimized, self.config.clip_grad_norm, error_if_nonfinite=True
            )
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.model.zero_grad(set_to_none=True)
        return True

    def _autocast(self):
        if self.config.precision == "fp32":
            return nullcontext()
        dtype = self.torch.float16 if self.config.precision == "fp16" else self.torch.bfloat16
        return self.torch.autocast(device_type=self.device.type, dtype=dtype)

    def _weight(self, batch):
        count = getattr(self.step, "normalization_count", lambda batch: 1.0)(batch)
        if not math.isfinite(count) or count <= 0:
            raise ValueError("Loss normalization requires a finite positive weight")
        return count

    def _validate(self, data):
        previous = self.model.training
        self.model.eval()
        total, count = 0.0, 0.0
        try:
            with self.torch.inference_mode():
                for batch in self._loader(data, 0, False):
                    batch = move(batch, self.device)
                    with self._autocast():
                        value = self.step(self.model, batch)
                    if not isinstance(value, self.torch.Tensor) or value.ndim != 0:
                        raise ValueError("Validation step must return a scalar loss tensor")
                    if not self.torch.isfinite(value).all().item():
                        raise ValueError("Non-finite validation loss")
                    weight = self._weight(batch)
                    total += value.item() * weight
                    count += weight
        finally:
            self.model.train(previous)
        if not count:
            raise ValueError("Validation dataset/loader is empty")
        return total if getattr(self.step, "reduction", "mean") == "sum" else total / count


def train(model, data, task="regression", validation=None, resume=None, **config):
    """Convenience training; use Trainer for custom optimizers, steps and callbacks."""
    return Trainer(model, task=task, config=TrainingConfig(**config)).fit(data, validation, resume)
