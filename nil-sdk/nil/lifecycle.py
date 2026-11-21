"""Local runs, immutable artifact copies and a transactional version registry."""

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".nil-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value):
        raise ValueError("Use a name of 1–100 letters, digits, dots, underscores or hyphens")
    return value


class Run:
    """One local run with machine-readable events suitable for IDE consumption."""

    def __init__(self, root="./nil_runs", name="training", config=None):
        self.id = f"run-{uuid.uuid4().hex}"
        self.path = Path(root).resolve() / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = {
            "schema_version": 1,
            "id": self.id,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "created",
            "config": config or {},
        }
        execution = os.environ.get("_NIL_EXECUTION_DIRECTORY")
        if execution:
            self.metadata["execution_job"] = Path(execution).name
        atomic_json(self.path / "run.json", self.metadata)
        if execution:
            atomic_json(Path(execution) / "runs" / f"{self.id}.json", {"path": str(self.path)})

    def event(self, kind, data=None, step=None):
        record = {
            "type": kind,
            "time": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "data": data or {},
        }
        encoded = json.dumps(record, allow_nan=False)
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")

    def status(self, state, **details):
        if state not in {"created", "running", "completed", "cancelled", "failed"}:
            raise ValueError("Unknown run status")
        self.metadata.update(status=state, **details)
        atomic_json(self.path / "run.json", self.metadata)
        self.event(f"job.{state}", details)

    def log(self, metrics, step):
        values = {k: float(v) for k, v in metrics.items()}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Metrics must be finite numbers")
        self.event("metrics", values, step)

    def artifact(self, source, name=None, **metadata):
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError("Artifacts must be regular files, not symlinks")
        name = _name(name or source.name)
        target = self.path / "artifacts" / name
        target.parent.mkdir(exist_ok=True)
        # Exclusive create prevents accidental replacement of a registered artifact.
        try:
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        except FileExistsError:
            raise
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        record = {
            **metadata,
            "name": name,
            "path": str(target),
            "sha256": sha256(target),
            "bytes": target.stat().st_size,
        }
        self.event("artifact.created", record)
        return record


class Registry:
    """Versioned local file artifacts. No executable model deserialization."""

    def __init__(self, root="./nil_registry"):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS models (
                name TEXT NOT NULL, version INTEGER NOT NULL, record TEXT NOT NULL,
                PRIMARY KEY(name, version))""")

    def _connect(self):
        return sqlite3.connect(self.root / "registry.sqlite", timeout=30)

    def register(self, name, file, metadata=None):
        name = _name(name)
        source = Path(file)
        if source.is_symlink() or not source.is_file():
            raise ValueError("Provide a regular artifact file, not a symlink")
        # Serialize metadata before mutating storage.
        json.dumps(metadata or {}, allow_nan=False)
        target = None
        created = False
        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                version = conn.execute(
                    "SELECT COALESCE(MAX(version),0)+1 FROM models WHERE name=?", (name,)
                ).fetchone()[0]
                target = self.root / "artifacts" / name / str(version) / source.name
                target.parent.mkdir(parents=True, exist_ok=False)
                created = True
                shutil.copyfile(source, target)
                sidecar = Path(str(source) + ".json")
                if sidecar.is_file() and not sidecar.is_symlink():
                    manifest = json.loads(sidecar.read_text())
                    if manifest.get("sha256") != sha256(target):
                        raise ValueError("Source checkpoint manifest does not match artifact")
                    atomic_json(str(target) + ".json", manifest)
                record = {
                    "schema_version": 1,
                    "name": name,
                    "version": version,
                    "path": str(target),
                    "sha256": sha256(target),
                    "bytes": target.stat().st_size,
                    "metadata": metadata or {},
                }
                conn.execute(
                    "INSERT INTO models VALUES (?,?,?)",
                    (name, version, json.dumps(record, allow_nan=False)),
                )
            return record
        except BaseException:
            if created:
                shutil.rmtree(target.parent)
            raise

    def resolve(self, name, version=None):
        _name(name)
        with self._connect() as conn:
            if version is None:
                row = conn.execute(
                    "SELECT record FROM models WHERE name=? ORDER BY version DESC LIMIT 1", (name,)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT record FROM models WHERE name=? AND version=?", (name, version)
                ).fetchone()
        if row is None:
            raise KeyError(f"No registered model {name!r}, version={version!r}")
        record = json.loads(row[0])
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("Artifact integrity check failed")
        return record

    def load(self, name, model, version=None):
        """Restore a NIL checkpoint into an explicitly constructed native model."""
        from nil.training.checkpoint import load

        record = self.resolve(name, version)
        load(record["path"], model)
        return model


def export_mlflow(result, tracking_uri, experiment="nil"):
    """Explicitly export a finished local run to an MLflow file/server backend.

    Calling this function opts into the specified backend's network/storage use.
    No implicit default service, credentials or model deserialization is used.
    """
    from nil.runtime import require

    client = require("mlflow.tracking", "tracking").MlflowClient(tracking_uri=tracking_uri)
    existing = client.get_experiment_by_name(experiment)
    experiment_id = existing.experiment_id if existing else client.create_experiment(experiment)
    remote = client.create_run(experiment_id, tags={"nil.run_id": result.run.id})
    run_id = remote.info.run_id
    try:
        for key, value in result.run.metadata["config"].items():
            client.log_param(run_id, key, value)
        for row in result.history:
            for key, value in row.items():
                if key != "epoch":
                    client.log_metric(run_id, key, float(value), step=row["epoch"])
        client.log_artifacts(run_id, str(result.run.path))
        client.set_terminated(run_id, "FINISHED" if result.status == "completed" else "KILLED")
    except BaseException:
        client.set_terminated(run_id, "FAILED")
        raise
    return run_id
