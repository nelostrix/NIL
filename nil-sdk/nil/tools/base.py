"""
Tool — the base class and decorator for defining agent tools.

Users can define tools in two ways:

1. Class-based:
    class SearchTool(Tool):
        name = "search"
        description = "Search the web for information"
        parameters = {"query": {"type": "string", "description": "Search query"}}

        def invoke(self, arguments):
            return do_search(arguments["query"])

2. Decorator-based:
    @tool
    def search(query: str) -> str:
        \"\"\"Search the web for information.\"\"\"
        return do_search(query)
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_type_hints


@dataclass
class Tool:
    """
    A tool that an agent can invoke.

    Attributes:
        name: Unique identifier for this tool.
        description: What this tool does (sent to the LLM).
        parameters: JSON Schema dict describing the tool's inputs.
        fn: The actual callable to execute.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    fn: Optional[Callable[..., Any]] = None

    def invoke(self, arguments: dict[str, Any]) -> Any:
        """
        Execute the tool with the given arguments.

        Override this in subclasses, or let the @tool decorator
        set self.fn automatically.
        """
        if self.fn is not None:
            # Filter arguments to only those the fucntions accepts
            sig = inspect.signature(self.fn)
            valid_args = {k: v for k, v in arguments.items() if k in sig.parameters}
            return self.fn(**valid_args)
        raise NotImplementedError(
            f"tool '{self.name}'has no implementation."
            "Either override invoke() or use the @tool decorator."
        )

    def to_schema(self) -> dict[str, Any]:
        """Convert to the LLM func
        tion-calling schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": [
                        k for k, v in self.parameters.items() if not v.get("optional", False)
                    ],
                },
            },
        }


# Python type  -> JSON schema type mapping
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool(fn: Callable[..., Any]) -> Tool:
    """
    Decorator that converts a function into a Tool.

    The function's name becomes the tool name, its docstring
    becomes the description, and type hints become the parameter schema.

    Example:
        @tool
        def lookup_user(user_id: str, include_email: bool = False) -> dict:
            \"\"\"Look up a user by their ID.\"\"\"
            return {"id": user_id, "email": "..."}
    """
    name = fn.__name__
    description = (fn.__doc__ or "").strip()

    # build parameter schema from type hints
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    parameters: dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_type = hints.get(param_name, str)
        json_type = _TYPE_MAP.get(param_type, "string")

        param_schema: dict[str, Any] = {"type": json_type}

        # if parameter has a defualt, it's Optional
        if param.default is not inspect.Parameter.empty:
            param_schema["optional"] = True
            param_schema["default"] = param.default
        parameters[param_name] = param_schema

    return Tool(
        name=name,
        description=description,
        parameters=parameters,
        fn=fn,
    )
