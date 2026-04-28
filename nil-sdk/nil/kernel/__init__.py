"""
Kernel — the agent execution engine.

The kernel handles the actual run loop: sending prompts to the LLM,
executing tool calls, recording every step in the tracer, and
producing a RunOutcome.
"""

from nil.kernel.outcome import RunOutcome
from nil.kernel.runner import KernelRunner
from nil.kernel.tool_manager import ToolManager
from nil.kernel.tracer import Tracer

__all__ = ["KernelRunner", "Tracer", "RunOutcome", "ToolManager"]
