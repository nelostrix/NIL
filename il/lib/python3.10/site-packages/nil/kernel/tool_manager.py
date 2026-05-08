"""
ToolManager — registers, validates, and executes tools.

The ToolManager sits between the LLM and the actual tool functions.
It converts Tool objects into the schema format LLMs expect,
dispatches calls, and captures results with timing.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from nil.tools.base import Tool

logger = logging.getLogger("nil.kernel.tool_manager")


class ToolManager:
    """
    Manages the set of tools available to an agent.

    Responsibilities:
        - Convert Tool objects to LLM-compatible schemas
        - Dispatch tool calls by name
        - Measure execution time and capture errors
    """

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises if duplicate name."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")

        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Optional[Tool]:
        """Look up a tool by name."""
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def to_schemas(self) -> list[dict[str, Any]]:
        """
        Convert all tools to the schema format expected by LLM APIs.

        Returns a list of tool definitions compatible with the
        Anthropic/OpenAI function-calling format.
        """
        schemas = []
        for tool in self._tools.values():
            schemas.append(tool.to_schema())
        return schemas

    def execute(self, name: str, arguments: dict[str, Any]):
        """
        Execute a tool by name with the given arguments.

        Returns:
            Dict with keys: output, duration_ms, error
        """
        tool = self._tools.get(name)
        if tool is None:
            return {
                "output": None,
                "duration_ms": 0.0,
                "error": f"tool '{name}'not found. Available names->{self.names}",
            }
        start = time.perf_counter()
        try:
            result = tool.invoke(arguments)
            elapsed = (time.perf_counter() - start) * 1000
            return {
                "output": result,
                "duration_ms": round(elapsed, 2),
                "error": None,
            }
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Tool '%s' failed: %s", name, str(e))
            return {
                "output": None,
                "duration_ms": round(elapsed, 2),
                "error": str(e),
            }

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
