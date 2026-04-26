"""
Nil is an SDK that lets developers build AI agents which autonomously
solve domain-specific problems and get better every time they run.

Quick start:
    import nil

    domain = nil.Domain(
        name="my-domain",
        tools=[my_tool],
        objectives=["solve X autonomously"],
        success_signal=lambda run: run.success
    )

    agent = nil.Agent(
        domain=domain,
        base_model="claude-sonnet-4-20250514",
        memory=nil.PersistentMemory("./agent_store"),
    )

    result = agent.run("Do the thing")
"""

__version__ = "0.0.1"

from nil.agent import Agent
from nil.config import EvolutionConfig, NilConfig
from nil.domain import Domain
from nil.memory.store import PersistantMemory
from nil.tools.base import Tool, tool

__all__ = [
    "Domain",
    "Agent",
    "EvolutionConfig",
    "NilConfig",
    "PersistentMemory",
    "Tool",
    "tool",
]
