"""Public help coverage and offline access, including extension-owned Markdown."""

import ast
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

import nil
from nil.documentation import topics


def test_every_public_module_and_catalog_model_has_help():
    source = Path(__file__).resolve().parents[1] / "nil"
    public_modules = {p.stem for p in source.glob("*.py") if not p.stem.startswith("_")}
    public_modules |= {p.name for p in source.iterdir() if p.is_dir() and not p.name.startswith("_")}
    available = set(topics())
    assert public_modules | set(nil.list_models()) <= available
    for topic in available:
        content = nil.help(topic)
        for heading in ("## Usage", "## Parameters", "## Output", "## Requirements", "## Limitations"):
            assert heading in content, (topic, heading)
        for snippet in re.findall(r"```python\n(.*?)```", content, flags=re.S):
            ast.parse(snippet, filename=f"{topic}/help.md")


def test_help_is_available_without_compute_backends_or_side_effects(tmp_path):
    source = str(Path(__file__).resolve().parents[1])
    script = """
import nil, sys, pathlib
from nil.documentation import topics
assert 'linear_regression' in topics()
assert 'fit_intercept' in nil.help('linear_regression')
assert 'classification' in nil.help('logistic_regression')
assert not any(name.split('.')[0] in {'torch','numpy','sklearn','xgboost'} for name in sys.modules)
assert not list(pathlib.Path('.').iterdir())
"""
    subprocess.run([sys.executable, "-S", "-c", script], cwd=tmp_path,
                   env={**os.environ, "PYTHONPATH": source}, check=True)


def test_help_rejects_unknown_and_path_topics():
    for topic in ("../runtime", "/etc/passwd", "", None):
        with pytest.raises(ValueError, match="topic"):
            nil.help(topic)
    with pytest.raises(ValueError, match="nil help --list"):
        nil.help("unknown_help_topic")


def test_custom_model_can_supply_its_own_help_file(tmp_path, monkeypatch):
    monkeypatch.setattr(nil.catalog, "_EXTENSIONS", {})
    help_file = tmp_path / "help.md"
    help_file.write_text("# custom_forecaster\n\nUse window=24.\n", encoding="utf-8")
    calls = []

    def factory(**parameters):
        calls.append(parameters)
        return parameters

    nil.register_model("custom_forecaster", factory, group="time_series", help_path=help_file)
    assert "window=24" in nil.help("custom_forecaster")
    assert "custom_forecaster" in topics()
    assert not calls
    assert nil.model("custom_forecaster", window=24) == {"window": 24}


def test_cli_reads_help_offline_and_reports_invalid_topics(tmp_path):
    source = str(Path(__file__).resolve().parents[1])
    options = {"cwd": tmp_path, "env": {**os.environ, "PYTHONPATH": source},
               "text": True, "capture_output": True}
    command = [sys.executable, "-S", "-m", "nil", "help"]
    result = subprocess.run([*command, "linear_regression"], check=True, **options)
    assert "fit_intercept" in result.stdout
    listed = subprocess.run([*command, "--list"], check=True, **options)
    assert "training" in listed.stdout and "linear_regression" in listed.stdout
    missing = subprocess.run([*command, "missing_topic"], **options)
    assert missing.returncode == 2
    assert "Unknown help topic" in missing.stderr
