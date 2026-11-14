"""State-dict checkpoints, integrity checked before restricted deserialization."""

import json
import os
import tempfile
from pathlib import Path

from nil.lifecycle import atomic_json, sha256
from nil.runtime import require


def save(path, model, optimizer=None, scheduler=None, **state):
    torch = require("torch", "torch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "state": state,
    }
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-")
    os.close(fd)
    try:
        with open(temp, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        atomic_json(
            str(path) + ".json",
            {"schema_version": 1, "sha256": sha256(path), "bytes": path.stat().st_size},
        )
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return path


def load(path, model, optimizer=None, scheduler=None, restore_rng=False):
    torch = require("torch", "torch")
    path = Path(path)
    manifest = json.loads(Path(str(path) + ".json").read_text())
    if manifest.get("schema_version") != 1 or sha256(path) != manifest["sha256"]:
        raise ValueError("Checkpoint integrity check failed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        if payload["optimizer"] is None:
            raise ValueError("Checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload["scheduler"] is None:
            raise ValueError("Checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        torch.set_rng_state(payload["rng"])
        if torch.cuda.is_available() and payload["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["state"]
