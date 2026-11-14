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
import math
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union, Literal, get_type_hints, get_origin, get_args


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
        self.validate(arguments)
        if self.fn is not None:
            return self.fn(**arguments)
        raise NotImplementedError(
            f"tool '{self.name}'has no implementation."
            "Either override invoke() or use the @tool decorator."
        )

    def validate(self, arguments: dict[str, Any]) -> None:
        _validate(arguments, self.to_schema()["function"]["parameters"], self.name)

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
                    "properties": {k: {a: b for a, b in v.items() if a != "optional"}
                                   for k, v in self.parameters.items()},
                    "additionalProperties": False,
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


def _schema(annotation):
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        return {}
    if annotation is type(None):
        return {"type": "null"}
    if origin in (Union, types.UnionType):
        return {"anyOf": [_schema(arg) for arg in args]}
    if origin is Literal:
        return {"enum": list(args)}
    if origin in (list, tuple) or annotation in (list, tuple):
        if origin is tuple and args and args[-1] is not Ellipsis:
            return {"type": "array", "prefixItems": [_schema(arg) for arg in args],
                    "minItems": len(args), "maxItems": len(args)}
        return {"type": "array", "items": _schema(args[0]) if args else {}}
    if origin is dict or annotation is dict:
        if args and args[0] not in (str, Any):
            raise TypeError("Tool dictionaries require string keys")
        return {"type": "object", "additionalProperties": _schema(args[1]) if args else {}}
    if annotation not in _TYPE_MAP:
        raise TypeError(f"Unsupported tool annotation: {annotation!r}")
    return {"type": _TYPE_MAP[annotation]}


def _validate(value, schema, path):
    if "anyOf" in schema:
        for choice in schema["anyOf"]:
            try:
                _validate(value, choice, path)
                return
            except ValueError:
                continue
        raise ValueError(f"{path}: value does not match any allowed type")
    if "enum" in schema and not any(type(value) is type(v) and value == v for v in schema["enum"]):
        raise ValueError(f"{path}: value is not an allowed choice")
    kind = schema.get("type")
    checks = {"string": lambda v: isinstance(v, str),
              "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
              "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
              "boolean": lambda v: isinstance(v, bool), "object": lambda v: isinstance(v, dict),
              "array": lambda v: isinstance(v, (list, tuple)), "null": lambda v: v is None}
    if kind in checks and not checks[kind](value):
        raise ValueError(f"{path}: expected {kind}")
    if kind == "object":
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path}.{key}: required argument missing")
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: object keys must be strings")
            if key in properties:
                _validate(item, properties[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise ValueError(f"{path}.{key}: unknown argument")
            elif isinstance(schema.get("additionalProperties"), dict):
                _validate(item, schema["additionalProperties"], f"{path}.{key}")
    if kind == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            raise ValueError(f"{path}: wrong number of items")
        prefix = schema.get("prefixItems", [])
        for index, item in enumerate(value):
            _validate(item, prefix[index] if index < len(prefix) else schema.get("items", {}),
                      f"{path}[{index}]")


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
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
                          inspect.Parameter.POSITIONAL_ONLY):
            raise TypeError("Tool parameters must be named, explicit keyword-compatible arguments")
        param_schema = _schema(param_type)

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
