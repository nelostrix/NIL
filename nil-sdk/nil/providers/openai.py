"""
OpenAIProvider — adapter for OpenAI's Chat Completions API.

Translates Nil's internal message format to OpenAI's API format.
Also works with any OpenAI-compatible API (Ollama, vLLM, etc.)
by setting OPENAI_BASE_URL.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from nil.config import NilConfig

logger = logging.getLogger("nil.providers.openai")


class OpenAIProvider:
    """
    LLM provider for OpenAI-compatible APIs.

    Requires the `openai` package: pip install openai
    Requires OPENAI_API_KEY environment variable.
    Optionally set OPENAI_BASE_URL for local models (Ollama, etc.)
    """

    def __init__(self, model: str, config: NilConfig) -> None:
        self.model = model
        self.config = config
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required. "
                    "Install it with: pip install openai"
                )
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Send a chat request to an OpenAI-compatible API.

        Returns a normalized response dict.
        """
        client = self._get_client()

        api_messages = [{"role": "system", "content": system}]
        for msg in messages:
            role = msg.get("role", "user")
            if role == "tool":
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }
                )
            elif role == "assistant" and msg.get("tool_calls"):
                api_messages.append(
                    {
                        "role": "assistant",
                        "content": msg.get("content", ""),
                        "tool_calls": [
                            {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name", ""),
                                    "arguments": __import__("json").dumps(
                                        tc.get("arguments", {})
                                    ),
                                },
                            }
                            for tc in msg["tool_calls"]
                        ],
                    }
                )
            else:
                api_messages.append(
                    {
                        "role": role,
                        "content": msg.get("content", ""),
                    }
                )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    def _parse_response(self, response: Any) -> dict[str, Any]:
        """Parse OpenAI response into normalized format."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": __import__("json").loads(tc.function.arguments),
                    }
                )

        return {
            "content": message.content,
            "tool_calls": tool_calls,
            "tokens_in": response.usage.prompt_tokens if response.usage else 0,
            "tokens_out": response.usage.completion_tokens if response.usage else 0,
        }
