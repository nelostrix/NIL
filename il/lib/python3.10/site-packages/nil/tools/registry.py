"""
Tool Registry — discover and load tools.

Phase 2+ feature: tools can be registered globally and discovered
by domain. For now, provides a simple in-process registry.
"""

from __future__ import annotations

from typing import Optional

from nil.tools.base import Tool

_GLOBAL_REGISTRY: dict[str, Tool] = {}


def register_tool(tool: Tool) -> None:
    """Register a tool in the global registry."""
    _GLOBAL_REGISTRY[tool.name] = tool


def get_tool(name: str) -> Optional[Tool]:
    """Look up a tool by name."""
    return _GLOBAL_REGISTRY.get(name)


def list_tools() -> list[str]:
    """List all registered tool names."""
    return list(_GLOBAL_REGISTRY.keys())
