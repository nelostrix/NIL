"""Artifact rejection paths must work before importing a model backend/deserializer."""
import hashlib
import json
import platform

import pytest

import nil


def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    payload = b"not executable model data"
    (root / "model.joblib").write_bytes(payload)
    manifest = {"schema_version": 1, "format": "nil-classical-joblib",
                "sha256": hashlib.sha256(payload).hexdigest(), "python": platform.python_version(),
                "dependencies": {"scikit-learn": "impossible-version"}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, manifest


def test_untrusted_load_is_rejected_before_accessing_any_path(tmp_path):
    with pytest.raises(ValueError, match="trusted=True"):
        nil.load(tmp_path / "does-not-exist")


def test_corruption_is_rejected_before_loading_native_python_data(tmp_path):
    root, _ = bundle(tmp_path)
    (root / "model.joblib").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        nil.load(root, trusted=True)


def test_version_mismatch_is_rejected_before_deserialization(tmp_path):
    root, _ = bundle(tmp_path)
    with pytest.raises(ValueError, match="version mismatch"):
        nil.load(root, trusted=True)


@pytest.mark.parametrize("manifest", [[], {"schema_version": True},
    {"schema_version": 1, "format": "nil-classical-joblib", "python": None, "dependencies": []}])
def test_malformed_manifests_have_clear_errors(tmp_path, manifest):
    root, _ = bundle(tmp_path)
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema|metadata"):
        nil.load(root, trusted=True)


def test_symlink_bundle_is_rejected(tmp_path):
    root, _ = bundle(tmp_path)
    link = tmp_path / "link"
    try:
        link.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="symlinks"):
        nil.load(link, trusted=True)
