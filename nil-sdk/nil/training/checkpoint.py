"""State-dict checkpoints, integrity checked before restricted deserialization."""

import json
import os
import platform
import shutil
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
        "model_structure": [(name, _class_name(module), module.extra_repr()) for name, module in model.named_modules()],
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "state": state,
        "optimizer_class": _class_name(optimizer),
        "optimizer_parameter_names": _optimizer_names(model, optimizer),
        "scheduler_class": _class_name(scheduler),
        "environment": {"torch": str(torch.__version__), "python": platform.python_version()},
    }
    if path.is_symlink() or Path(str(path) + ".json").is_symlink():
        raise ValueError("Checkpoints must not use symlinks")
    previous = None
    if path.exists():
        previous = json.loads(Path(str(path) + ".json").read_text())
        if not isinstance(previous, dict) or sha256(path) != previous.get("sha256"):
            raise ValueError("Existing checkpoint is inconsistent; recover it before replacing")
        try:
            _generation(path, previous)
        except ValueError:
            fd, backup = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.generation-", suffix=".pt")
            os.close(fd)
            try:
                shutil.copyfile(path, backup)
                previous['artifact'] = Path(backup).name
                atomic_json(str(path) + '.json', previous)
            except BaseException:
                Path(backup).unlink(missing_ok=True)
                raise
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.generation-", suffix=".pt")
    os.close(fd)
    try:
        with open(temp, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        manifest = {"schema_version": 1, "sha256": sha256(temp), "bytes": Path(temp).stat().st_size,
                    "artifact": Path(temp).name}
        if previous:
            manifest['previous'] = {key: value for key, value in previous.items() if key != 'previous'}
        _publish_alias(Path(temp), path)
        # The manifest is the commit record. A crash before this commit can be
        # repaired from the prior immutable generation with recover().
        atomic_json(str(path) + '.json', manifest)
        _sync_directory(path.parent)
    except BaseException:
        committed = False
        try:
            committed = json.loads(Path(str(path) + '.json').read_text()).get('artifact') == Path(temp).name
        except (OSError, ValueError, AttributeError):
            pass
        if not committed:
            if previous:
                _publish_alias(_generation(path, previous), path)
            else:
                path.unlink(missing_ok=True)
            Path(temp).unlink(missing_ok=True)
        raise
    _prune_generations(path, manifest)
    return path


def _generation(path, manifest):
    name = manifest.get('artifact')
    if not isinstance(name, str) or Path(name).name != name or not name.startswith(f'.{path.name}.generation-'):
        raise ValueError("Checkpoint has no safe recoverable generation")
    source = path.parent / name
    if source.is_symlink() or not source.is_file() or sha256(source) != manifest.get('sha256'):
        raise ValueError("Checkpoint generation integrity check failed")
    return source


def _publish_alias(source, destination):
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix='.checkpoint-publish-')
    os.close(fd)
    os.unlink(temporary)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _sync_directory(path):
    if os.name == 'posix':
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass


def _prune_generations(path, manifest):
    keep = {manifest.get('artifact')}
    if isinstance(manifest.get('previous'), dict):
        keep.add(manifest['previous'].get('artifact'))
    for candidate in path.parent.iterdir():
        if candidate.name.startswith(f'.{path.name}.generation-') and candidate.name not in keep and candidate.is_file() and not candidate.is_symlink():
            candidate.unlink()


def recover(path, *, previous=False):
    """Restore the committed generation, or explicitly fall back one generation.

    Does not deserialize tensors. Use a single writer/recoverer for each path.
    """
    path = Path(path)
    metadata = Path(str(path) + '.json')
    if path.is_symlink() or metadata.is_symlink():
        raise ValueError("Checkpoints must not use symlinks")
    manifest = json.loads(metadata.read_text())
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
        raise ValueError("Unsupported checkpoint manifest")
    if previous:
        manifest = manifest.get('previous')
        if not isinstance(manifest, dict):
            raise ValueError("Checkpoint has no previous committed generation")
    if not previous and path.is_file() and sha256(path) == manifest.get('sha256'):
        _prune_generations(path, manifest)
        return path
    generation = _generation(path, manifest)
    _publish_alias(generation, path)
    atomic_json(metadata, manifest)
    _sync_directory(path.parent)
    _prune_generations(path, manifest)
    return path


def _class_name(value):
    return f"{type(value).__module__}.{type(value).__qualname__}" if value is not None else None


def _optimizer_names(model, optimizer):
    if optimizer is None:
        return None
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    if any(id(parameter) not in names for group in optimizer.param_groups for parameter in group['params']):
        raise ValueError('Optimizer contains parameters outside this model')
    return [[names[id(parameter)] for parameter in group['params']] for group in optimizer.param_groups]


def read(path, *, allow_version_mismatch=False):
    """Validate and deserialize without mutating model, optimizer or RNG state."""
    torch = require("torch", "torch")
    path = Path(path)
    if path.is_symlink() or Path(str(path) + ".json").is_symlink():
        raise ValueError("Checkpoints must not use symlinks")
    manifest = json.loads(Path(str(path) + ".json").read_text())
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1 or sha256(path) != manifest.get("sha256"):
        raise ValueError("Checkpoint integrity check failed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1 or not isinstance(payload.get("model"), dict) or not isinstance(payload.get("state"), dict):
        raise ValueError("Unsupported checkpoint schema")
    environment = payload.get("environment")
    if environment is not None and not allow_version_mismatch:
        if not isinstance(environment, dict) or not isinstance(environment.get("python"), str) or environment.get("torch") != str(torch.__version__) or environment["python"].split('.')[:2] != platform.python_version().split('.')[:2]:
            raise ValueError("Checkpoint Python/torch environment mismatch")
    return payload


def restore(payload, model, optimizer=None, scheduler=None, restore_rng=False):
    """Preflight standard state compatibility before loading any model weights."""
    torch = require("torch", "torch")
    current = model.state_dict()
    saved = payload["model"]
    structure = [(name, _class_name(module), module.extra_repr()) for name, module in model.named_modules()]
    if payload.get('model_structure') is not None and payload['model_structure'] != structure:
        raise ValueError("Checkpoint model structure mismatch")
    if set(current) != set(saved):
        raise ValueError("Checkpoint model keys do not match")
    for key, expected in current.items():
        actual = saved[key]
        if isinstance(expected, torch.Tensor):
            if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise ValueError(f"Checkpoint model shape/dtype mismatch at {key}")
        elif type(actual) is not type(expected):
            raise ValueError(f"Checkpoint model state type mismatch at {key}")
    for label, component in (("optimizer", optimizer), ("scheduler", scheduler)):
        if component is not None:
            if payload.get(label) is None:
                raise ValueError(f"Checkpoint has no {label} state")
            saved_class = payload.get(label + "_class")
            if saved_class is not None and saved_class != _class_name(component):
                raise ValueError(f"Checkpoint {label} class mismatch")
    if optimizer is not None:
        if payload.get('optimizer_parameter_names') is not None and payload['optimizer_parameter_names'] != _optimizer_names(model, optimizer):
            raise ValueError('Checkpoint optimizer parameter names/order mismatch')
        groups = payload["optimizer"].get("param_groups", [])
        current_groups = optimizer.state_dict()["param_groups"]
        if len(groups) != len(current_groups) or any(len(first["params"]) != len(second["params"]) for first, second in zip(groups, current_groups)):
            raise ValueError("Checkpoint optimizer parameter groups mismatch")
    if restore_rng:
        rng = payload.get("rng")
        if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.shape != torch.get_rng_state().shape:
            raise ValueError("Checkpoint RNG state is incompatible")
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


def load(path, model, optimizer=None, scheduler=None, restore_rng=False, *, allow_version_mismatch=False):
    payload = read(path, allow_version_mismatch=allow_version_mismatch)
    return restore(payload, model, optimizer, scheduler, restore_rng)
