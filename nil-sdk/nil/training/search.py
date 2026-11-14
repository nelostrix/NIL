"""Bounded sequential grid search over training configuration fields."""

import math
from dataclasses import asdict, dataclass, fields
from itertools import product
from pathlib import Path

from nil.lifecycle import Run, atomic_json
from nil.runtime import seed
from nil.training.checkpoint import load
from nil.training.trainer import Trainer, TrainingConfig


@dataclass
class SearchResult:
    best_config: dict
    best_checkpoint: Path
    trials: list
    run: Run

    def load_best(self, model):
        load(self.best_checkpoint, model)
        return model


def grid_search(
    model_factory,
    data,
    validation,
    parameters,
    *,
    config=None,
    task="regression",
    metric="val_loss",
    mode="min",
    max_trials=20,
):
    """Train fresh models with reproducible initialization and select a final-epoch metric.

    parameters varies TrainingConfig fields (e.g. learning_rate, batch_size).
    The factory takes no arguments. This is sequential local execution, not a
    distributed scheduler. Model/data factories must not share mutable state.
    """
    if validation is None:
        raise ValueError("Provide held-out validation data for model selection")
    if mode not in {"min", "max"}:
        raise ValueError("mode must be min or max")
    if isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials < 1:
        raise ValueError("max_trials must be a positive integer")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("Provide a nonempty parameter grid")
    allowed = {f.name for f in fields(TrainingConfig)} - {"output_dir", "name", "seed"}
    if set(parameters) - allowed:
        raise ValueError(f"Unsupported search fields: {sorted(set(parameters) - allowed)}")
    if any(not isinstance(values, (list, tuple)) or not values for values in parameters.values()):
        raise ValueError("Each search field requires a nonempty list or tuple")
    size = math.prod(len(values) for values in parameters.values())
    if size > max_trials:
        raise ValueError(f"Grid contains {size} trials, exceeding max_trials={max_trials}")
    base = asdict(config or TrainingConfig())
    # Validate all configurations before any run/model is created.
    candidates = []
    for values in product(*parameters.values()):
        candidate = {**base, **dict(zip(parameters, values))}
        TrainingConfig(**candidate)
        candidates.append(candidate)
    search = Run(
        base["output_dir"], "grid-search", {"grid": parameters, "metric": metric, "mode": mode}
    )
    search.status("running")
    trials, best = [], None
    try:
        for index, candidate in enumerate(candidates, 1):
            child_config = {
                **candidate,
                "output_dir": str(search.path / "trials"),
                "name": f"trial-{index}",
            }
            try:
                seed(candidate["seed"])
                trainer = Trainer(model_factory(), task=task, config=TrainingConfig(**child_config))
                result = trainer.fit(data, validation=validation)
                if result.status != "completed":
                    raise ValueError(f"Trial did not complete: {result.status}")
                value = float(result.history[-1][metric])
                if not math.isfinite(value):
                    raise ValueError("Selection metric must be finite")
                trial = {
                    "trial": index,
                    "config": candidate,
                    "status": "completed",
                    "value": value,
                    "run": str(result.run.path),
                    "checkpoint": str(result.checkpoint),
                }
                if best is None or (
                    value < best["value"] if mode == "min" else value > best["value"]
                ):
                    best = trial
            except Exception as exc:
                trial = {
                    "trial": index,
                    "config": candidate,
                    "status": "failed",
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            trials.append(trial)
            search.event("search.trial", trial, index)
            atomic_json(search.path / "search.json", {"trials": trials, "best": best})
        if best is None:
            raise ValueError("All search trials failed; inspect search.json and trial run events")
        search.status("completed", best_trial=best["trial"], best_value=best["value"])
        return SearchResult(best["config"], Path(best["checkpoint"]), trials, search)
    except BaseException as exc:
        search.status("failed", error=type(exc).__name__, message=str(exc))
        raise
