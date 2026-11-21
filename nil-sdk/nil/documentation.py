"""Offline Markdown help; no model backend or documentation server is needed."""

from importlib.resources import files
from pathlib import Path


def topics():
    """List bundled help topics and explicitly registered custom models."""
    from nil.catalog import _EXTENSIONS

    root = files("nil").joinpath("_help")
    names = {item.name for item in root.iterdir() if item.joinpath("help.md").is_file()}
    return sorted(names | set(_EXTENSIONS))


def read_help(topic="catalog"):
    """Return a topic's help.md text. Call print(nil.help(topic)) in a terminal."""
    if not isinstance(topic, str) or not topic.isidentifier():
        raise ValueError("Help topic must be a name such as linear_regression or training")
    from nil.catalog import _EXTENSIONS

    if topic in _EXTENSIONS:
        _group, factory, help_path = _EXTENSIONS[topic]
        if help_path is not None:
            return Path(help_path).read_text(encoding="utf-8")
        description = getattr(factory, "__doc__", None) or "No help file was supplied."
        return f"# {topic}\n\n{description}\n\nRegister with help_path= to attach help.md.\n"
    resource = files("nil").joinpath("_help", topic, "help.md")
    if not resource.is_file():
        raise ValueError(f"Unknown help topic {topic!r}. Run nil help --list to see available topics.")
    return resource.read_text(encoding="utf-8")
