"""Small native callbacks; snapshots and stop decisions use real epoch metrics."""

import math

from nil.training.checkpoint import save


class EarlyStopping:
    """Stop after patience epochs without improvement; save best model weights.

    The callback resets for each new run. Its best.pt is a model snapshot, not
    a training-resume checkpoint; resume training from the trainer's latest.pt.
    State is intentionally reset when starting a resumed run as well.
    """

    def __init__(self, patience=3, monitor="val_loss", mode="min", min_delta=0.0):
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError("patience must be a positive integer")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be min or max")
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        self.patience, self.monitor, self.mode, self.min_delta = patience, monitor, mode, min_delta
        self._run_id = None
        self.best_value = None
        self.best_checkpoint = None
        self.bad_epochs = 0

    def __call__(self, trainer, run, metrics):
        if self._run_id != run.id:
            self._run_id = run.id
            self.best_value, self.best_checkpoint, self.bad_epochs = None, None, 0
        if self.monitor not in metrics:
            raise ValueError(f"Early stopping metric {self.monitor!r} is absent")
        value = float(metrics[self.monitor])
        if not math.isfinite(value):
            raise ValueError("Early stopping requires finite metrics")
        improved = self.best_value is None or (
            value < self.best_value - self.min_delta
            if self.mode == "min"
            else value > self.best_value + self.min_delta
        )
        if improved:
            self.best_value, self.bad_epochs = value, 0
            self.best_checkpoint = save(
                run.path / "checkpoints" / "best.pt",
                trainer.model,
                epoch=metrics["epoch"],
                metric=self.monitor,
                value=value,
            )
            run.event(
                "checkpoint.best",
                {"path": str(self.best_checkpoint), "metric": self.monitor, "value": value},
                metrics["epoch"],
            )
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                trainer.finish("early_stopping")
