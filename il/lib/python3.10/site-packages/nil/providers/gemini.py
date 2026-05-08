"""
GeminiProvider — adapter for Google's Gemini API.

Translates Nil's internal message format to Gemini's API format
using the google-genai SDK.

Requires: pip install google-genai
Requires: GEMINI_API_KEY environment variable
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from nil.config import NilConfig

logger = logging.getLogger("nil.providers.gemini")


class GeminiProvider:
    """
    LLM provider for Google's Gemini models.

    Requires the `google-genai` package: pip install google-genai
    Requires GEMINI_API_KEY environment variable.

    Supported models: gemini-2.0-flash, gemini-2.5-pro, etc.
    """

    def __init__(self, model: str = "gemini-2.0-flash", config: NilConfig = None) -> None:
        self.model = model
        self.config = config or NilConfig()
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "The 'google-genai' package is required. "
                    "Install it with: pip install google-genai"
                )
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set.")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Send a chat request to Gemini.

        Returns a normalized response dict:
            {
                "content": str or None,
                "tool_calls": list of {name, arguments, id},
                "tokens_in": int,
                "tokens_out": int,
            }
        """
        client = self._get_client()
        from google.genai import types

        # Build contents from messages
        contents = self._convert_messages(messages)

        # Build config with system instruction
        gen_config = types.GenerateContentConfig(
            system_instruction=system,
        )

        # Add tools if provided
        if tools:
            gen_config.tools = [self._convert_tools(tools)]

        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=gen_config,
        )

        return self._parse_response(response)

    def _convert_messages(self, messages: list[dict]) -> list[Any]:
        """Convert internal message format to Gemini contents."""
        from google.genai import types

        contents = []
        for msg in messages:
            role = msg.get("role", "user")

            if role == "tool":
                # Tool result → function response
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=msg.get("tool_name", "function"),
                                    response={"result": msg.get("content", "")},
                                )
                            )
                        ],
                    )
                )

            elif role == "assistant" and msg.get("tool_calls"):
                # Assistant with tool calls → model role with function calls
                parts = []
                if msg.get("content"):
                    parts.append(types.Part(text=msg["content"]))
                for tc in msg["tool_calls"]:
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=tc.get("name", ""),
                                args=tc.get("arguments", {}),
                            )
                        )
                    )
                contents.append(types.Content(role="model", parts=parts))

            elif role == "assistant":
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=msg.get("content", ""))],
                    )
                )

            else:
                # User message
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=msg.get("content", ""))],
                    )
                )

        return contents

    def _convert_tools(self, tools: list[dict]) -> Any:
        """Convert tools from Nil format to Gemini function declarations."""
        from google.genai import types

        declarations = []
        for t in tools:
            func = t.get("function", t)
            params = func.get("parameters", {})

            # Convert parameter schema to Gemini format
            properties = {}
            required = []
            for param_name, param_schema in params.get("properties", {}).items():
                prop_type = param_schema.get("type", "STRING").upper()
                # Map JSON Schema types to Gemini types
                type_map = {
                    "STRING": "STRING",
                    "INTEGER": "INTEGER",
                    "NUMBER": "NUMBER",
                    "BOOLEAN": "BOOLEAN",
                    "ARRAY": "ARRAY",
                    "OBJECT": "OBJECT",
                }
                properties[param_name] = types.Schema(
                    type=type_map.get(prop_type, "STRING"),
                    description=param_schema.get("description", ""),
                )
                if not param_schema.get("optional", False):
                    required.append(param_name)

            schema = types.Schema(
                type="OBJECT",
                properties=properties,
                required=required if required else None,
            )

            declarations.append(
                types.FunctionDeclaration(
                    name=func.get("name", ""),
                    description=func.get("description", ""),
                    parameters=schema,
                )
            )

        return types.Tool(function_declarations=declarations)

    def _parse_response(self, response: Any) -> dict[str, Any]:
        """Parse Gemini response into normalized format."""
        content = ""
        tool_calls = []

        candidate = response.candidates[0] if response.candidates else None
        if candidate and candidate.content:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    content += part.text
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        {
                            "id": f"gemini_call_{fc.name}",
                            "name": fc.name,
                            "arguments": dict(fc.args) if fc.args else {},
                        }
                    )

        # Token usage
        tokens_in = 0
        tokens_out = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_in = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            tokens_out = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        return {
            "content": content if content else None,
            "tool_calls": tool_calls,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
