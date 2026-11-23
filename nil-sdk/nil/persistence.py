"""Local bundles of complete native classical pipelines, schema and versions."""
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

from nil._classical_common import is_classical, unwrap
from nil.lifecycle import sha256
from nil.runtime import require


def save(model, path):
    """Save a fitted pipeline to a new directory; existing bundles are preserved."""
    model = unwrap(model)
    if not is_classical(model):
        raise TypeError("nil.save currently supports classical estimators; neural models use checkpoints")
    require("sklearn.utils.validation", "classical").check_is_fitted(model)
    joblib = require("joblib", "classical")
    destination = Path(path).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".nil-model-", dir=destination.parent))
    try:
        artifact = temporary / "model.joblib"
        joblib.dump(model, artifact, compress=0, protocol=5)
        versions = {}
        packages = ["scikit-learn", "numpy", "scipy", "joblib"]
        parts = [model, *model.get_params(deep=True).values()]
        modules = {type(part).__module__.split('.')[0] for part in parts}
        for module, distribution in {"nil": "nil-sdk", "xgboost": "xgboost",
                                     "imblearn": "imbalanced-learn", "lightgbm": "lightgbm",
                                     "catboost": "catboost"}.items():
            if module in modules:
                packages.append(distribution)
        for part in parts:
            packages.extend(getattr(part, '_nil_backend_distributions', ()))
        if "xgboost" in modules:
            packages.append("xgboost-cpu")
        for package in packages:
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                pass
        manifest = {
            "schema_version": 1, "format": "nil-classical-joblib", "python": platform.python_version(),
            "dependencies": versions, "sha256": sha256(artifact),
            "estimator": f"{type(model).__module__}.{type(model).__name__}",
            "feature_count": int(model.n_features_in_) if hasattr(model, "n_features_in_") else None,
            "feature_names": model.feature_names_in_.tolist() if hasattr(model, "feature_names_in_") else None,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        # Reserve exclusively and publish the manifest last (portable across OSes).
        destination.mkdir()
        try:
            os.replace(artifact, destination / "model.joblib")
            os.replace(temporary / "manifest.json", destination / "manifest.json")
        except BaseException:
            (destination / "model.joblib").unlink(missing_ok=True)
            try:
                destination.rmdir()
            except OSError:
                pass
            raise
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load(path, *, trusted=False, allow_version_mismatch=False):
    """Load trusted joblib artifacts. Hashes check integrity, not authenticity."""
    if not trusted:
        raise ValueError("Loading native Python artifacts requires trusted=True for a source you trust")
    root = Path(path)
    artifact, manifest_file = root / "model.joblib", root / "manifest.json"
    if root.is_symlink() or artifact.is_symlink() or manifest_file.is_symlink():
        raise ValueError("Model bundles must not use symlinks")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1 or manifest.get("format") != "nil-classical-joblib":
        raise ValueError("Unsupported model bundle schema or format")
    dependencies = manifest.get("dependencies")
    if not isinstance(manifest.get("python"), str) or not isinstance(dependencies, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in dependencies.items()):
        raise ValueError("Malformed model environment metadata")
    if sha256(artifact) != manifest.get("sha256"):
        raise ValueError("Model artifact integrity check failed")
    if not allow_version_mismatch:
        if platform.python_version().split('.')[:2] != manifest.get("python", "").split('.')[:2]:
            raise ValueError("Python versions differ; recreate the original model environment")
        for package, expected in dependencies.items():
            try:
                actual = metadata.version(package)
            except metadata.PackageNotFoundError:
                actual = None
            if actual != expected:
                raise ValueError(f"Dependency version mismatch: {package} expected {expected}, found {actual}")
    return require("joblib", "classical").load(artifact)
