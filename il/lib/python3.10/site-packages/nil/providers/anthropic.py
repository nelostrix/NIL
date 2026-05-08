"""
AnthropicProvider — adapter for the Anthropic Messages API.

Translates Nil's internal message format to Anthropic's API format
and back.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from nil.config import NilConfig

logger = logging.getLogger("nil.providers.anthropic")


class AnthropicProvider:
    """
    LLM provider for Anthropic's Claude models.

    Requires the `anthropic` package: pip install anthropic
    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, model: str, config: NilConfig) -> None:
        self.model = model
        self.config = config
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required. "
                    "Install it with: pip install anthropic"
                )
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Send a chat request to Claude.

        Returns a normalized response dict:
            {
                "content": str or None,
                "tool_calls": list of {name, arguments, id},
                "tokens_in": int,
                "tokens_out": int,
            }
        """
        client = self._get_client()

        # Convert messages to Anthropic format
        api_messages = self._convert_messages(messages)

        # Build request
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": api_messages,
        }

        # Convert tools to Anthropic format
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = client.messages.create(**kwargs)

        return self._parse_response(response)

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        """Convert internal message format to Anthropic format."""
        api_msgs = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "tool":
                # Anthropic uses tool_result content blocks
                api_msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id", ""),
                                "content": msg.get("content", ""),
                            }
                        ],
                    }
                )
            elif role == "assistant" and msg.get("tool_calls"):
                # Assistant message with tool use
                content_blocks = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc.get("name", ""),
                            "input": tc.get("arguments", {}),
                        }
                    )
                api_msgs.append({"role": "assistant", "content": content_blocks})
            else:
                api_msgs.append(
                    {
                        "role": role,
                        "content": msg.get("content", ""),
                    }
                )
        return api_msgs

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert tools from OpenAI-style to Anthropic format."""
        anthropic_tools = []
        for t in tools:
            func = t.get("function", t)
            anthropic_tools.append(
                {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )
        return anthropic_tools

    def _parse_response(self, response: Any) -> dict[str, Any]:
        """Parse Anthropic response into normalized format."""
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    }
                )

        return {
            "content": content if content else None,
            "tool_calls": tool_calls,
            "tokens_in": response.usage.input_tokens,
            "tokens_out": response.usage.output_tokens,
        }
