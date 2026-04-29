"""
KernelRunner — the core agent execution loop.

This is where the agent actually runs. The loop is:
    1. Build the prompt (system + task + context + skills)
    2. Call the LLM
    3. If the LLM wants to use a tool → execute it, feed result back
    4. Repeat until the LLM produces a final answer or max steps hit
    5. Return a RunOutcome

The runner is model-agnostic — it delegates actual API calls to
a provider adapter (see nil/providers/).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from nil.config import NilConfig
from nil.domain import Domain
from nil.kernel.outcome import RunOutcome, StepRecord
from nil.kernel.tool_manager import ToolManager
from nil.kernel.tracer import Tracer
from nil.memory.store import PersistentMemory

logger = logging.getLogger("nil.kernel.runner")

# Maximum agentic loop iterations to prevent runaway
MAX_STEPS = 25


class KernelRunner:
    """
    Executes an agent task through an LLM + tool loop.

    This is the heart of Phase 1. It connects:
        - The LLM (via a provider adapter)
        - The tools (via ToolManager)
        - The tracer (for experience capture)
        - The memory (for retrieving learned skills)
    """

    def __init__(
        self,
        domain: Domain,
        model: str,
        config: NilConfig,
        memory: PersistentMemory,
        provider: Optional[str] = None,
    ) -> None:
        self.domain = domain
        self.model = model
        self.config = config
        self.memory = memory
        self.tool_manager = ToolManager(domain.tools)

        # Lazily load the provider
        self._provider = self._init_provider(provider)

    def _init_provider(self, provider_name: Optional[str] = None) -> Any:
        """
        Initialize the LLM provider adapter.

        Returns a provider instance with a `chat()` method.
        Falls back to a stub provider for testing if no API key is found.
        """
        name = provider_name or self._detect_provider()

        if name == "anthropic":
            from nil.providers.anthropic import AnthropicProvider

            return AnthropicProvider(model=self.model, config=self.config)
        elif name == "openai":
            from nil.providers.openai import OpenAIProvider

            return OpenAIProvider(model=self.model, config=self.config)
        elif name == "gemini":
            from nil.providers.gemini import GeminiProvider

            return GeminiProvider(model=self.model, config=self.config)
        else:
            from nil.providers.stub import StubProvider

            logger.warning(
                "No LLM provider detected. Using StubProvider for testing. "
                "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY to use a real model."
            )
            return StubProvider(model=self.model, config=self.config)

    def _detect_provider(self) -> str:
        """Auto-detect which LLM provider to use based on env vars."""
        import os

        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            return "openai"
        elif os.environ.get("GEMINI_API_KEY"):
            return "gemini"
        return "stub"

    def _build_system_prompt(self) -> str:
        """
        Construct the system prompt from:
          - Domain config (name, objectives)
          - Active strategy (reasoning hints, avoidance rules, etc.)
          - Learned skills (compressed action sequences)
          - World model knowledge (constraints, tool guidance)
          - Available tools
        """
        parts = [
            f"You are an AI agent specialized in the domain: {self.domain.name}.",
            "",
            "Your objectives:",
        ]
        for obj in self.domain.objectives:
            parts.append(f"  - {obj}")

        # Inject active strategy
        strategy = self.memory.get_active_strategy(self.domain.name)
        if strategy:
            # Reasoning hints
            hints = strategy.get("reasoning_hints", [])
            if hints:
                parts.append("")
                parts.append("Guidelines from past experience:")
                for hint in hints:
                    parts.append(f"  - {hint}")

            # Avoidance rules
            avoidance = strategy.get("avoidance_rules", [])
            if avoidance:
                parts.append("")
                parts.append("Rules to follow:")
                for rule in avoidance:
                    parts.append(f"  - {rule}")

            # Error handling hints
            error_hints = strategy.get("error_handling_hints", [])
            if error_hints:
                parts.append("")
                parts.append("Known error patterns to handle:")
                for hint in error_hints:
                    parts.append(f"  - {hint}")

            # Preferred first tool
            first_tool = strategy.get("preferred_first_tool")
            if first_tool:
                parts.append(f"\nRecommended: start with '{first_tool}' when applicable.")

            # Skill injections from strategy
            skill_inj = strategy.get("skill_injections", [])
            if skill_inj:
                parts.append("")
                parts.append("Proven approaches:")
                for s in skill_inj:
                    parts.append(f"  - {s}")

        # Inject learned skills from memory
        skills = self.memory.list_skills(self.domain.name)
        if skills:
            parts.append("")
            parts.append("Learned skills:")
            for skill_name in skills[:10]:  # Cap to avoid prompt overflow
                skill_data = self.memory.get_skill(self.domain.name, skill_name)
                if skill_data and skill_data.get("description"):
                    desc = skill_data["description"]
                    # Use first 200 chars of description
                    parts.append(f"  - {desc[:200]}")
                else:
                    parts.append(f"  - {skill_name}")

        # Available tools
        parts.append("")
        parts.append(
            "Available tools: " + ", ".join(self.tool_manager.names)
            if self.tool_manager.names
            else "No tools available."
        )
        parts.append("")
        parts.append(
            "Think step by step. Use tools when needed. Provide a clear final answer when done."
        )

        return "\n".join(parts)

    def execute(
        self,
        task: str,
        context: dict[str, Any],
        tracer: Tracer,
    ) -> RunOutcome:
        """
        Run the agent loop for a given task.

        Returns a RunOutcome with all steps, timing, and the final output.
        """
        start_time = time.time()
        system_prompt = self._build_system_prompt()
        messages = []

        # Add context if provided
        if context:
            task_with_context = f"{task}\n\nAdditional context:\n{context}"
        else:
            task_with_context = task

        messages.append({"role": "user", "content": task_with_context})

        outcome = RunOutcome(
            run_id=tracer.run_id,
            task=task,
        )

        tool_schemas = self.tool_manager.to_schemas()

        for step in range(MAX_STEPS):
            # Call the LLM
            llm_start = time.perf_counter()
            try:
                response = self._provider.chat(
                    system=system_prompt,
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                )
            except Exception as e:
                elapsed = (time.perf_counter() - llm_start) * 1000
                tracer.record_llm_call(
                    prompt=messages[-1],
                    response=None,
                    duration_ms=elapsed,
                    error=str(e),
                )
                outcome.error = str(e)
                break

            llm_elapsed = (time.perf_counter() - llm_start) * 1000

            tracer.record_llm_call(
                prompt=messages[-1].get("content", "")[:200],
                response=response.get("content", "")[:200]
                if isinstance(response.get("content"), str)
                else str(response.get("content", ""))[:200],
                tokens_in=response.get("tokens_in", 0),
                tokens_out=response.get("tokens_out", 0),
                duration_ms=llm_elapsed,
            )

            # Check if the LLM wants to call a tool
            tool_calls = response.get("tool_calls", [])

            if not tool_calls:
                # No tool calls — this is the final answer
                outcome.output = response.get("content", "")
                break

            # Process each tool call
            messages.append(
                {
                    "role": "assistant",
                    "content": response.get("content", ""),
                    "tool_calls": tool_calls,
                }
            )

            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})

                result = self.tool_manager.execute(tool_name, tool_args)

                tracer.record_tool_call(
                    name=tool_name,
                    tool_input=tool_args,
                    tool_output=result["output"],
                    duration_ms=result["duration_ms"],
                    error=result["error"],
                )

                # Feed tool result back to LLM
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": str(result["output"])
                        if result["output"]
                        else f"Error: {result['error']}",
                    }
                )

                # Record step in outcome
                outcome.steps.append(
                    StepRecord(
                        step_number=len(outcome.steps) + 1,
                        step_type="tool_call",
                        tool_name=tool_name,
                        tool_input=tool_args,
                        tool_output=result["output"],
                        duration_ms=result["duration_ms"],
                        error=result["error"],
                        timestamp=time.time(),
                    )
                )
        else:
            # Hit MAX_STEPS
            outcome.error = f"Agent exceeded maximum steps ({MAX_STEPS})"
            logger.warning("Run %s hit max steps", tracer.run_id)

        outcome.duration = time.time() - start_time
        outcome.total_tokens_in = sum(s.tokens_in for s in tracer._steps)
        outcome.total_tokens_out = sum(s.tokens_out for s in tracer._steps)

        return outcome
