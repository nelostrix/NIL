"""
Textual Gradient Engine — the core intellectual property of Nil.

Inspired by TextGrad (Yuksekgonul et al., 2024), this module treats
agent optimization as automatic differentiation via text.

The core idea:
    1. Take a failed (or suboptimal) execution trace
    2. Feed it to an LLM with structured critique instructions
    3. The LLM produces a "textual gradient" — structured feedback
       identifying what went wrong and proposing specific fixes
    4. Apply the gradient to rewrite the agent's strategy
       (system prompt, tool ordering, reasoning hints, skill injections)

What makes this different from DSPy:
    - DSPy optimizes at compile-time. This runs continuously at runtime.
    - DSPy needs labeled training examples. This learns from the agent's
      own experience traces — no human labels required.
    - DSPy optimizes prompts. This optimizes entire strategies including
      tool ordering, skill injection, and reasoning chain structure.

What makes this different from NeoCognition:
    - NeoCognition builds proprietary agents. This is an SDK that lets
      any developer make their agents self-improving.
    - This is open and composable — works with any LLM, any tool set.

The gradient pipeline:
    Trace → Critique → Gradient → Mutation → Validation → Deploy

Each stage is independently testable and deterministic when using
the StubGradientProvider (for testing without API keys).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from nil.evolution.analyzer import (
    AnalysisReport,
    CausalPattern,
    FailureCluster,
    StepAttribution,
    TraceAnalyzer,
)
from nil.memory.store import PersistentMemory

logger = logging.getLogger("nil.evolution.gradient")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class GradientType(str, Enum):
    """Types of textual gradients the engine can produce."""

    PROMPT_REWRITE = "prompt_rewrite"
    TOOL_REORDER = "tool_reorder"
    SKILL_INJECTION = "skill_injection"
    REASONING_HINT = "reasoning_hint"
    FAILURE_AVOIDANCE = "failure_avoidance"
    COST_OPTIMIZATION = "cost_optimization"


@dataclass
class TextualGradient:
    """
    A single textual gradient — a structured improvement proposal.

    This is the fundamental unit of the evolution engine. Each gradient
    targets one specific aspect of the agent's strategy.
    """

    gradient_id: str
    gradient_type: GradientType
    critique: str  # What went wrong (LLM's analysis)
    proposal: str  # What to change (LLM's suggestion)
    target: str  # What part of the strategy to modify
    confidence: float = 0.0  # 0-1, how confident the LLM is
    source_traces: list[str] = field(default_factory=list)
    reasoning: str = ""  # Chain-of-thought behind the proposal
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gradient_id": self.gradient_id,
            "gradient_type": self.gradient_type.value,
            "critique": self.critique,
            "proposal": self.proposal,
            "target": self.target,
            "confidence": self.confidence,
            "source_traces": self.source_traces,
            "reasoning": self.reasoning,
            "created_at": self.created_at,
        }


@dataclass
class StrategyDelta:
    """
    A concrete change to apply to the agent's strategy.

    Produced by applying a TextualGradient to the current strategy.
    """

    delta_id: str
    source_gradient: str  # gradient_id
    mutation_type: str
    changes: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    expected_impact: float = 0.0
    rollback_data: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_id": self.delta_id,
            "source_gradient": self.source_gradient,
            "mutation_type": self.mutation_type,
            "changes": self.changes,
            "description": self.description,
            "expected_impact": self.expected_impact,
            "created_at": self.created_at,
        }


@dataclass
class GradientBatch:
    """A batch of gradients produced from one analysis cycle."""

    batch_id: str
    domain: str
    gradients: list[TextualGradient] = field(default_factory=list)
    deltas: list[StrategyDelta] = field(default_factory=list)
    analysis_summary: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def gradient_count(self) -> int:
        return len(self.gradients)

    @property
    def has_actionable_deltas(self) -> bool:
        return len(self.deltas) > 0


# ---------------------------------------------------------------------------
# Gradient Provider Interface
# ---------------------------------------------------------------------------


class GradientProvider:
    """
    Base class for gradient providers.

    A gradient provider takes trace data and produces textual critiques.
    Subclass this to use different LLMs or local models.
    """

    def critique_traces(
        self,
        failed_traces: list[dict[str, Any]],
        successful_traces: list[dict[str, Any]],
        domain_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Generate critiques for failed traces by comparing with successes.

        Returns a list of structured critique dicts.
        """
        raise NotImplementedError

    def propose_improvements(
        self,
        critiques: list[dict[str, Any]],
        current_strategy: dict[str, Any],
        domain_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Given critiques, propose concrete strategy improvements.

        Returns a list of structured improvement proposals.
        """
        raise NotImplementedError


class StubGradientProvider(GradientProvider):
    """
    Deterministic gradient provider for testing.

    Produces predictable gradients based on trace analysis
    without calling any LLM API. Uses rule-based heuristics
    that mirror what a real LLM critique would produce.
    """

    def critique_traces(
        self,
        failed_traces: list[dict[str, Any]],
        successful_traces: list[dict[str, Any]],
        domain_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        critiques = []

        # Heuristic 1: Compare tool sequences
        success_seqs = [tuple(t.get("tool_sequence", [])) for t in successful_traces]
        fail_seqs = [tuple(t.get("tool_sequence", [])) for t in failed_traces]

        # Find tools in successes but missing from failures
        success_tools = set()
        for seq in success_seqs:
            success_tools.update(seq)
        fail_tools = set()
        for seq in fail_seqs:
            fail_tools.update(seq)

        missing_in_failures = success_tools - fail_tools
        extra_in_failures = fail_tools - success_tools

        if missing_in_failures:
            critiques.append(
                {
                    "type": "missing_tool",
                    "critique": (
                        f"Failed traces are missing tools that successful traces use: "
                        f"{', '.join(sorted(missing_in_failures))}. "
                        f"The agent should be instructed to use these tools."
                    ),
                    "severity": "high",
                    "affected_tools": sorted(missing_in_failures),
                    "confidence": 0.8,
                }
            )

        if extra_in_failures:
            critiques.append(
                {
                    "type": "harmful_tool",
                    "critique": (
                        f"Failed traces use tools not seen in successes: "
                        f"{', '.join(sorted(extra_in_failures))}. "
                        f"These tools may be counterproductive."
                    ),
                    "severity": "medium",
                    "affected_tools": sorted(extra_in_failures),
                    "confidence": 0.6,
                }
            )

        # Heuristic 2: Check for common errors
        error_types = set()
        for t in failed_traces:
            err = t.get("error", "")
            if err:
                error_types.add(err[:50])
            for step in t.get("steps", []):
                if step.get("error"):
                    error_types.add(step["error"][:50])

        if error_types:
            critiques.append(
                {
                    "type": "recurring_error",
                    "critique": (
                        f"Recurring errors detected: {'; '.join(sorted(error_types))}. "
                        f"The agent should be given specific guidance to avoid these."
                    ),
                    "severity": "high",
                    "error_patterns": sorted(error_types),
                    "confidence": 0.7,
                }
            )

        # Heuristic 3: Tool ordering
        if success_seqs and fail_seqs:
            # Check if successful sequences have a consistent first tool
            first_tools_success = [seq[0] for seq in success_seqs if seq]
            first_tools_fail = [seq[0] for seq in fail_seqs if seq]
            if first_tools_success and first_tools_fail:
                from collections import Counter

                s_first = Counter(first_tools_success).most_common(1)
                f_first = Counter(first_tools_fail).most_common(1)
                if s_first and f_first and s_first[0][0] != f_first[0][0]:
                    critiques.append(
                        {
                            "type": "wrong_start",
                            "critique": (
                                f"Successful traces typically start with '{s_first[0][0]}' "
                                f"but failed traces start with '{f_first[0][0]}'. "
                                f"The agent should be instructed to begin with '{s_first[0][0]}'."
                            ),
                            "severity": "medium",
                            "recommended_start": s_first[0][0],
                            "confidence": 0.7,
                        }
                    )

        # Heuristic 4: Sequence length
        if success_seqs and fail_seqs:
            avg_success_len = sum(len(s) for s in success_seqs) / len(success_seqs)
            avg_fail_len = sum(len(s) for s in fail_seqs) / len(fail_seqs)
            if avg_fail_len > avg_success_len * 1.5:
                critiques.append(
                    {
                        "type": "over_tooling",
                        "critique": (
                            f"Failed traces use too many tool calls "
                            f"(avg {avg_fail_len:.1f} vs {avg_success_len:.1f} in successes). "
                            f"The agent may be overthinking or retrying unnecessarily."
                        ),
                        "severity": "low",
                        "confidence": 0.5,
                    }
                )

        # Always produce at least one critique if there are failures
        if not critiques and failed_traces:
            critiques.append(
                {
                    "type": "general",
                    "critique": (
                        f"Agent failed on {len(failed_traces)} traces. "
                        f"No specific pattern detected — may need more data."
                    ),
                    "severity": "low",
                    "confidence": 0.3,
                }
            )

        return critiques

    def propose_improvements(
        self,
        critiques: list[dict[str, Any]],
        current_strategy: dict[str, Any],
        domain_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        proposals = []

        for critique in critiques:
            ctype = critique.get("type", "general")

            if ctype == "missing_tool":
                tools = critique.get("affected_tools", [])
                proposals.append(
                    {
                        "gradient_type": GradientType.REASONING_HINT.value,
                        "target": "system_prompt",
                        "proposal": (
                            f"Add reasoning hint: 'Always consider using "
                            f"{', '.join(tools)} before concluding.'"
                        ),
                        "confidence": critique.get("confidence", 0.5),
                        "changes": {
                            "reasoning_hints": [
                                f"Always consider using {', '.join(tools)} before concluding."
                            ],
                        },
                    }
                )

            elif ctype == "harmful_tool":
                tools = critique.get("affected_tools", [])
                proposals.append(
                    {
                        "gradient_type": GradientType.FAILURE_AVOIDANCE.value,
                        "target": "system_prompt",
                        "proposal": (
                            f"Add avoidance rule: 'Avoid using {', '.join(tools)} "
                            f"unless absolutely necessary.'"
                        ),
                        "confidence": critique.get("confidence", 0.5),
                        "changes": {
                            "avoidance_rules": [
                                f"Avoid using {', '.join(tools)} unless absolutely necessary."
                            ],
                        },
                    }
                )

            elif ctype == "recurring_error":
                errors = critique.get("error_patterns", [])
                proposals.append(
                    {
                        "gradient_type": GradientType.FAILURE_AVOIDANCE.value,
                        "target": "system_prompt",
                        "proposal": (f"Add error handling guidance for: {'; '.join(errors[:3])}"),
                        "confidence": critique.get("confidence", 0.5),
                        "changes": {
                            "error_handling_hints": [
                                f"Watch for and handle: {err}" for err in errors[:3]
                            ],
                        },
                    }
                )

            elif ctype == "wrong_start":
                tool = critique.get("recommended_start", "")
                if tool:
                    proposals.append(
                        {
                            "gradient_type": GradientType.TOOL_REORDER.value,
                            "target": "tool_ordering",
                            "proposal": f"Prioritize starting with '{tool}'.",
                            "confidence": critique.get("confidence", 0.5),
                            "changes": {
                                "preferred_first_tool": tool,
                            },
                        }
                    )

            elif ctype == "over_tooling":
                proposals.append(
                    {
                        "gradient_type": GradientType.COST_OPTIMIZATION.value,
                        "target": "system_prompt",
                        "proposal": (
                            "Add efficiency hint: 'Be concise in tool usage. "
                            "Avoid unnecessary retries.'"
                        ),
                        "confidence": critique.get("confidence", 0.5),
                        "changes": {
                            "reasoning_hints": [
                                "Be concise in tool usage. Avoid unnecessary retries. "
                                "If a tool call fails, analyze the error before retrying."
                            ],
                        },
                    }
                )

        return proposals


class AnthropicGradientProvider(GradientProvider):
    """
    Uses Claude to generate textual gradients.

    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self.model = model
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            import os

            try:
                import anthropic
            except ImportError:
                raise ImportError("pip install anthropic")
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def critique_traces(
        self,
        failed_traces: list[dict[str, Any]],
        successful_traces: list[dict[str, Any]],
        domain_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        client = self._get_client()

        # Build a compact representation of traces
        failed_summary = self._summarize_traces(failed_traces, "FAILED")
        success_summary = self._summarize_traces(successful_traces, "SUCCESSFUL")

        prompt = f"""You are an AI agent optimization expert. Analyze these execution traces and identify why some failed while others succeeded.

DOMAIN: {domain_context.get("name", "unknown")}
OBJECTIVES: {", ".join(domain_context.get("objectives", []))}
AVAILABLE TOOLS: {", ".join(domain_context.get("tools", []))}

{success_summary}

{failed_summary}

Produce a JSON array of critiques. Each critique must have:
- "type": one of "missing_tool", "harmful_tool", "recurring_error", "wrong_start", "over_tooling", "wrong_sequence", "missing_verification"
- "critique": detailed explanation of the problem
- "severity": "high", "medium", or "low"
- "confidence": 0.0-1.0
- "affected_tools": list of tool names involved (if applicable)
- "error_patterns": list of error strings (if applicable)
- "recommended_start": tool name (if type is "wrong_start")

Return ONLY the JSON array, no markdown, no explanation."""

        response = client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        # Parse JSON, stripping markdown fences if present
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM critique response: %s", text[:200])
            return [
                {
                    "type": "general",
                    "critique": text[:500],
                    "severity": "low",
                    "confidence": 0.3,
                }
            ]

    def propose_improvements(
        self,
        critiques: list[dict[str, Any]],
        current_strategy: dict[str, Any],
        domain_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        client = self._get_client()

        prompt = f"""You are an AI agent optimization expert. Given these critiques of an agent's performance, propose concrete improvements to its strategy.

DOMAIN: {domain_context.get("name", "unknown")}
CURRENT STRATEGY: {json.dumps(current_strategy, indent=2)[:1500]}

CRITIQUES:
{json.dumps(critiques, indent=2)[:2000]}

Produce a JSON array of improvement proposals. Each proposal must have:
- "gradient_type": one of "prompt_rewrite", "tool_reorder", "skill_injection", "reasoning_hint", "failure_avoidance", "cost_optimization"
- "target": what to modify ("system_prompt", "tool_ordering", "skill_list", "reasoning_chain")
- "proposal": plain English description of the change
- "confidence": 0.0-1.0
- "changes": object with the actual modifications to apply. Use keys like:
  - "reasoning_hints": list of hint strings to add to the prompt
  - "avoidance_rules": list of rules to avoid certain behaviors
  - "preferred_first_tool": tool name to start with
  - "skill_injections": list of skill descriptions
  - "prompt_additions": list of strings to append to system prompt
  - "error_handling_hints": list of error patterns to watch for

Return ONLY the JSON array, no markdown, no explanation."""

        response = client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM proposal response: %s", text[:200])
            return []

    def _summarize_traces(self, traces: list[dict], label: str) -> str:
        """Compact trace summary for LLM context."""
        if not traces:
            return f"--- {label} TRACES: none ---"

        lines = [f"--- {label} TRACES ({len(traces)} total) ---"]
        for t in traces[:10]:  # Cap to avoid context overflow
            seq = t.get("tool_sequence", [])
            err = t.get("error", "")
            cost = t.get("total_cost", 0)
            dur = t.get("total_duration", 0)
            task = t.get("task", "")[:80]
            lines.append(
                f"  Task: {task} | Tools: {' → '.join(seq) or 'none'} | "
                f"Cost: ${cost:.4f} | Duration: {dur:.1f}s"
                + (f" | Error: {err[:80]}" if err else "")
            )

            # Include step-level errors
            for step in t.get("steps", []):
                if step.get("error"):
                    lines.append(
                        f"    Step {step.get('step_number', '?')} "
                        f"({step.get('tool_name', '?')}): {step['error'][:60]}"
                    )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------


class TextualGradientEngine:
    """
    The Textual Gradient Engine — Nil's core evolution mechanism.

    Orchestrates the full gradient pipeline:
        1. Analyze traces (via TraceAnalyzer)
        2. Select traces for critique (worst failures, best successes)
        3. Generate critiques (via GradientProvider)
        4. Produce textual gradients
        5. Convert gradients to strategy deltas
        6. Return deltas for evaluation

    Usage:
        engine = TextualGradientEngine(memory)
        batch = engine.compute_gradients("my-domain")

        for delta in batch.deltas:
            print(delta.description)
            # Test and apply...

    With a real LLM:
        engine = TextualGradientEngine(
            memory,
            provider=AnthropicGradientProvider("claude-sonnet-4-20250514")
        )
    """

    def __init__(
        self,
        memory: PersistentMemory,
        provider: Optional[GradientProvider] = None,
        analyzer: Optional[TraceAnalyzer] = None,
        max_traces_per_critique: int = 10,
        min_failures_for_gradient: int = 3,
    ) -> None:
        self.memory = memory
        self.provider = provider or StubGradientProvider()
        self.analyzer = analyzer or TraceAnalyzer(memory)
        self.max_traces_per_critique = max_traces_per_critique
        self.min_failures_for_gradient = min_failures_for_gradient
        self._batch_counter = 0

    def compute_gradients(
        self,
        domain: str,
        max_traces: int = 200,
    ) -> GradientBatch:
        """
        Run the full gradient computation pipeline.

        Returns a GradientBatch containing textual gradients
        and their corresponding strategy deltas.
        """
        self._batch_counter += 1
        batch_id = f"gb-{domain}-{self._batch_counter}-{int(time.time())}"

        # Step 1: Get traces and run analysis
        traces = self.memory.get_traces(domain, limit=max_traces)
        if not traces:
            return GradientBatch(batch_id=batch_id, domain=domain)

        successes = [t for t in traces if t.get("outcome_success")]
        failures = [t for t in traces if not t.get("outcome_success")]

        if len(failures) < self.min_failures_for_gradient:
            logger.info(
                "Domain '%s' has only %d failures (need %d) — skipping gradient",
                domain,
                len(failures),
                self.min_failures_for_gradient,
            )
            return GradientBatch(
                batch_id=batch_id,
                domain=domain,
                analysis_summary={
                    "total_traces": len(traces),
                    "failures": len(failures),
                    "reason": "insufficient_failures",
                },
            )

        # Step 2: Run causal analysis
        report = self.analyzer.full_analysis(domain, max_traces=max_traces)

        # Step 3: Select traces for critique
        selected_failures = self._select_traces_for_critique(failures)
        selected_successes = self._select_traces_for_critique(successes)

        # Step 4: Build domain context
        domain_context = self._build_domain_context(domain, report)

        # Step 5: Generate critiques via provider
        critiques = self.provider.critique_traces(
            failed_traces=selected_failures,
            successful_traces=selected_successes,
            domain_context=domain_context,
        )

        # Step 6: Convert critiques to textual gradients
        gradients = self._critiques_to_gradients(critiques, selected_failures, batch_id)

        # Step 7: Get current strategy
        current_strategy = self.memory.get_active_strategy(domain) or {}

        # Step 8: Generate improvement proposals
        proposals = self.provider.propose_improvements(
            critiques=critiques,
            current_strategy=current_strategy,
            domain_context=domain_context,
        )

        # Step 9: Convert proposals to strategy deltas
        deltas = self._proposals_to_deltas(proposals, gradients, batch_id)

        # Step 10: Enrich deltas with analysis-driven improvements
        analysis_deltas = self._analysis_to_deltas(report, batch_id)
        deltas.extend(analysis_deltas)

        batch = GradientBatch(
            batch_id=batch_id,
            domain=domain,
            gradients=gradients,
            deltas=deltas,
            analysis_summary={
                "total_traces": report.total_traces,
                "successes": report.total_successes,
                "failures": report.total_failures,
                "baseline_rate": report.baseline_success_rate,
                "critiques_generated": len(critiques),
                "gradients_produced": len(gradients),
                "deltas_produced": len(deltas),
                "significant_patterns": len(report.significant_patterns),
                "systemic_failures": len(report.systemic_failures),
            },
        )

        logger.info(
            "Gradient batch %s: %d critiques → %d gradients → %d deltas",
            batch_id,
            len(critiques),
            len(gradients),
            len(deltas),
        )

        return batch

    # ------------------------------------------------------------------
    # Internal pipeline stages
    # ------------------------------------------------------------------

    def _select_traces_for_critique(self, traces: list[dict]) -> list[dict]:
        """
        Select the most informative traces for critique.

        Prioritizes:
        - Most recent traces (temporal relevance)
        - Highest cost failures (most wasteful)
        - Diverse error types (coverage)
        """
        if len(traces) <= self.max_traces_per_critique:
            return traces

        # Sort by recency (newest first)
        sorted_traces = sorted(traces, key=lambda t: t.get("created_at", 0), reverse=True)

        # Take a mix: half most recent, half highest cost
        half = self.max_traces_per_critique // 2
        recent = sorted_traces[:half]

        cost_sorted = sorted(traces, key=lambda t: t.get("total_cost", 0), reverse=True)
        high_cost = []
        seen_ids = {t["run_id"] for t in recent}
        for t in cost_sorted:
            if t["run_id"] not in seen_ids:
                high_cost.append(t)
                seen_ids.add(t["run_id"])
            if len(high_cost) >= self.max_traces_per_critique - half:
                break

        return recent + high_cost

    def _build_domain_context(self, domain: str, report: AnalysisReport) -> dict[str, Any]:
        """Build context dict for the gradient provider."""
        skills = self.memory.list_skills(domain)
        return {
            "name": domain,
            "objectives": [],  # Populated from Domain object if available
            "tools": list({a.tool_name for a in report.step_attributions}),
            "skills": skills,
            "baseline_success_rate": report.baseline_success_rate,
            "total_traces": report.total_traces,
            "positive_tools": [a.tool_name for a in report.positive_tools],
            "negative_tools": [a.tool_name for a in report.negative_tools],
            "redundant_tools": report.redundant_tools,
        }

    def _critiques_to_gradients(
        self,
        critiques: list[dict],
        source_traces: list[dict],
        batch_id: str,
    ) -> list[TextualGradient]:
        """Convert raw critiques into TextualGradient objects."""
        gradients = []
        source_ids = [t.get("run_id", "") for t in source_traces[:10]]

        for i, critique in enumerate(critiques):
            ctype = critique.get("type", "general")

            # Map critique type to gradient type
            gradient_type_map = {
                "missing_tool": GradientType.REASONING_HINT,
                "harmful_tool": GradientType.FAILURE_AVOIDANCE,
                "recurring_error": GradientType.FAILURE_AVOIDANCE,
                "wrong_start": GradientType.TOOL_REORDER,
                "over_tooling": GradientType.COST_OPTIMIZATION,
                "wrong_sequence": GradientType.TOOL_REORDER,
                "missing_verification": GradientType.SKILL_INJECTION,
                "general": GradientType.PROMPT_REWRITE,
            }

            grad = TextualGradient(
                gradient_id=f"{batch_id}-g{i}",
                gradient_type=gradient_type_map.get(ctype, GradientType.PROMPT_REWRITE),
                critique=critique.get("critique", ""),
                proposal=critique.get("critique", ""),  # Will be refined by proposals
                target=self._infer_target(ctype),
                confidence=critique.get("confidence", 0.5),
                source_traces=source_ids,
                reasoning=f"Identified via {ctype} analysis",
            )
            gradients.append(grad)

        return gradients

    def _proposals_to_deltas(
        self,
        proposals: list[dict],
        gradients: list[TextualGradient],
        batch_id: str,
    ) -> list[StrategyDelta]:
        """Convert improvement proposals into StrategyDelta objects."""
        deltas = []
        for i, prop in enumerate(proposals):
            # Link to closest gradient by type
            source_grad = ""
            prop_type = prop.get("gradient_type", "")
            for g in gradients:
                if g.gradient_type.value == prop_type:
                    source_grad = g.gradient_id
                    break
            if not source_grad and gradients:
                source_grad = gradients[0].gradient_id

            delta = StrategyDelta(
                delta_id=f"{batch_id}-d{i}",
                source_gradient=source_grad,
                mutation_type=prop.get("gradient_type", "prompt_rewrite"),
                changes=prop.get("changes", {}),
                description=prop.get("proposal", ""),
                expected_impact=prop.get("confidence", 0.5),
            )
            deltas.append(delta)

        return deltas

    def _analysis_to_deltas(self, report: AnalysisReport, batch_id: str) -> list[StrategyDelta]:
        """
        Generate deltas directly from the causal analysis report,
        without going through the LLM critique pipeline.

        This handles clear-cut improvements that don't need
        LLM reasoning: dead-end avoidance, skill injection
        from high-confidence patterns, etc.
        """
        deltas = []
        delta_idx = 100  # Start high to avoid ID collision

        # Delta from dead-end transitions
        dead_ends = report.dead_end_transitions
        if dead_ends:
            avoid_transitions = [
                f"After '{de.from_tool}', do not use '{de.to_tool}'" for de in dead_ends[:5]
            ]
            deltas.append(
                StrategyDelta(
                    delta_id=f"{batch_id}-d{delta_idx}",
                    source_gradient="analysis",
                    mutation_type="failure_avoidance",
                    changes={"avoidance_rules": avoid_transitions},
                    description=f"Avoid {len(dead_ends)} dead-end tool transitions",
                    expected_impact=0.4,
                )
            )
            delta_idx += 1

        # Delta from significant causal patterns
        sig_patterns = report.significant_patterns[:3]
        if sig_patterns:
            skill_hints = []
            for p in sig_patterns:
                seq_str = " → ".join(p.sequence)
                skill_hints.append(
                    f"Preferred sequence: {seq_str} (success rate: {p.success_rate:.0%})"
                )
            deltas.append(
                StrategyDelta(
                    delta_id=f"{batch_id}-d{delta_idx}",
                    source_gradient="analysis",
                    mutation_type="skill_injection",
                    changes={"skill_injections": skill_hints},
                    description=f"Inject {len(sig_patterns)} proven tool sequences as skills",
                    expected_impact=0.6,
                )
            )
            delta_idx += 1

        # Delta from systemic failures
        systemic = report.systemic_failures
        if systemic:
            error_hints = [
                f"Watch for: '{fc.error_signature[:80]}'"
                for fc in systemic[:3]
            ]
            deltas.append(
                StrategyDelta(
                    delta_id=f"{batch_id}-d{delta_idx}",
                    source_gradient="analysis",
                    mutation_type="failure_avoidance",
                    changes={"error_handling_hints": error_hints},
                    description=f"Add handling for {len(systemic)} systemic failure patterns",
                    expected_impact=0.5,
                )
            )

        return deltas

    @staticmethod
    def _infer_target(critique_type: str) -> str:
        """Infer which strategy component a critique targets."""
        return {
            "missing_tool": "system_prompt",
            "harmful_tool": "system_prompt",
            "recurring_error": "system_prompt",
            "wrong_start": "tool_ordering",
            "over_tooling": "system_prompt",
            "wrong_sequence": "tool_ordering",
            "missing_verification": "skill_list",
        }.get(critique_type, "system_prompt")

    # ------------------------------------------------------------------
    # Strategy application
    # ------------------------------------------------------------------

    def apply_deltas(
        self,
        domain: str,
        deltas: list[StrategyDelta],
    ) -> dict[str, Any]:
        """
        Merge a list of deltas into the current strategy and save.

        Returns the new strategy dict.
        """
        current = self.memory.get_active_strategy(domain) or {}
        merged = dict(current)

        for delta in deltas:
            changes = delta.changes
            for key, value in changes.items():
                if isinstance(value, list):
                    existing = merged.get(key, [])
                    if isinstance(existing, list):
                        # Deduplicate
                        for item in value:
                            if item not in existing:
                                existing.append(item)
                        merged[key] = existing
                    else:
                        merged[key] = value
                else:
                    merged[key] = value

        # Compute composite score
        avg_impact = sum(d.expected_impact for d in deltas) / len(deltas) if deltas else 0

        version = self.memory.store_strategy(
            domain=domain,
            strategy_data=merged,
            score=avg_impact,
        )

        logger.info(
            "Applied %d deltas to domain '%s' → strategy v%d (score=%.2f)",
            len(deltas),
            domain,
            version,
            avg_impact,
        )

        return merged
