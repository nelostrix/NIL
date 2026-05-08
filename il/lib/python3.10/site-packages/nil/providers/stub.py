"""
StubProvider — a fake LLM provider for testing.

Returns deterministic responses so you can test the kernel,
tracer, memory, and tool execution without burning API credits.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from nil.config import NilConfig

logger = logging.getLogger("nil.providers.stub")


class StubProvider:
    """
    Fake LLM provider for local testing.

    Behavior:
        - If tools are available, calls the first tool on the first call
        - Returns a summary response on subsequent calls
        - Tracks call count for deterministic behavior
    """

    def __init__(self, model: str, config: NilConfig) -> None:
        self.model = model
        self.config = config
        self._call_count = 0

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        self._call_count += 1
        last_msg = messages[-1] if messages else {}
        last_content = last_msg.get("content", "")

        # First call with tools: invoke the first tool
        if self._call_count == 1 and tools:
            first_tool = tools[0]
            func = first_tool.get("function", first_tool)
            tool_name = func.get("name", "unknown")
            # Build stub arguments from parameter schema
            props = func.get("parameters", {}).get("properties", {})
            stub_args = {}
            for k, v in props.items():
                ptype = v.get("type", "string")
                if ptype == "string":
                    stub_args[k] = f"stub_{k}_value"
                elif ptype == "integer":
                    stub_args[k] = 1
                elif ptype == "number":
                    stub_args[k] = 1.0
                elif ptype == "boolean":
                    stub_args[k] = True
                else:
                    stub_args[k] = f"stub_{k}"

            return {
                "content": f"I'll use the {tool_name} tool to help with this.",
                "tool_calls": [
                    {
                        "id": f"stub_call_{self._call_count}",
                        "name": tool_name,
                        "arguments": stub_args,
                    }
                ],
                "tokens_in": 100,
                "tokens_out": 50,
            }

        # Subsequent calls or no tools: return final answer
        self._call_count = 0  # Reset for next run
        return {
            "content": (
                f"[StubProvider] Task processed successfully. "
                f"Model: {self.model}. "
                f"Last input: {str(last_content)[:100]}"
            ),
            "tool_calls": [],
            "tokens_in": 80,
            "tokens_out": 40,
        }
