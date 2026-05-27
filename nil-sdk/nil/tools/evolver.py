"""
Tool Evolution — Agent discovers repeated workflows and compiles them into reusable tools.

Pipeline:
  1. WorkflowDetector mines traces for frequent tool sequences
  2. WorkflowCompiler converts sequences into callable Tool objects
  3. ToolEvolver orchestrates detection + compilation + registration
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from nil.tools.base import Tool
from nil.tools.registry import register_tool

logger = logging.getLogger("nil.tools.evolver")


@dataclass
class WorkflowStep:
    tool_name: str
    parameter_keys: list[str] = field(default_factory=list)
    parameter_defaults: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowSignature:
    steps: list[WorkflowStep]

    @property
    def normalized_key(self) -> str:
        return "\x00".join(s.tool_name for s in self.steps)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.normalized_key.encode()).hexdigest()[:16]


@dataclass
class WorkflowOccurrence:
    trace_id: str
    step_indices: tuple[int, int]
    success: bool
    parameters: list[dict[str, Any]]


@dataclass
class DetectedWorkflow:
    signature: WorkflowSignature
    occurrences: list[WorkflowOccurrence] = field(default_factory=list)

    @property
    def occurrence_count(self) -> int:
        return len(self.occurrences)

    @property
    def success_count(self) -> int:
        return sum(1 for o in self.occurrences if o.success)

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.occurrence_count, 1)

    @property
    def confidence(self) -> float:
        if self.occurrence_count < 3:
            return 0.0
        freq_score = min(1.0, self.occurrence_count / 20)
        return freq_score * 0.4 + self.success_rate * 0.6

    def parameter_schema(self) -> dict[str, Any]:
        param_values: dict[str, set] = defaultdict(set)
        for occ in self.occurrences:
            for step_params in occ.parameters:
                for k, v in step_params.items():
                    param_values[k].add(
                        json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                    )

        schema = {}
        for key, values in param_values.items():
            if len(values) <= 1:
                continue
            sample_values = [
                json.loads(v) if v.startswith(("{", "[")) else v
                for v in list(values)[:3]
            ]
            value_types = set(type(v) for v in sample_values)
            if bool in value_types:
                schema[key] = {"type": "boolean", "optional": True}
            elif int in value_types or all(
                isinstance(v, (int, float)) for v in sample_values
            ):
                schema[key] = {"type": "number", "optional": True}
            else:
                schema[key] = {"type": "string", "optional": True}

        return schema

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature.normalized_key,
            "steps": [
                {"tool_name": s.tool_name, "parameter_keys": s.parameter_keys}
                for s in self.signature.steps
            ],
            "occurrence_count": self.occurrence_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "confidence": self.confidence,
        }


class WorkflowDetector:
    def __init__(
        self,
        min_occurrences: int = 3,
        min_success_rate: float = 0.7,
        min_sequence_length: int = 2,
        max_sequence_length: int = 10,
    ):
        self.min_occurrences = min_occurrences
        self.min_success_rate = min_success_rate
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length

    def detect(self, traces: list[Any]) -> list[DetectedWorkflow]:
        tool_sequences = self._extract_tool_sequences(traces)
        candidates = self._mine_sequences(tool_sequences)
        return self._filter_and_merge(candidates)

    def _extract_tool_sequences(self, traces: list[Any]) -> list[dict]:
        result = []
        for trace in traces:
            if hasattr(trace, "tool_sequence"):
                seq = trace.tool_sequence
                params = getattr(trace, "tool_params", [{}] * len(seq))
                success = getattr(trace, "success", True)
                tid = getattr(
                    trace, "trace_id", getattr(trace, "run_id", str(id(trace)))
                )
            elif isinstance(trace, dict):
                seq = trace.get("tool_sequence", [])
                params = trace.get("tool_params", [{}] * len(seq))
                success = trace.get("success", trace.get("outcome_success", True))
                tid = trace.get("run_id", trace.get("trace_id", str(id(trace))))
            else:
                continue

            if len(seq) >= self.min_sequence_length:
                result.append(
                    {
                        "sequence": seq,
                        "params": params,
                        "trace_id": tid,
                        "success": bool(success),
                    }
                )
        return result

    def _mine_sequences(
        self, tool_sequences: list[dict]
    ) -> dict[str, DetectedWorkflow]:
        workflow_map: dict[str, DetectedWorkflow] = {}

        for entry in tool_sequences:
            seq = entry["sequence"]
            params = entry["params"]

            for length in range(
                self.min_sequence_length,
                min(len(seq), self.max_sequence_length) + 1,
            ):
                for i in range(len(seq) - length + 1):
                    sub_seq = seq[i : i + length]
                    sub_params = params[i : i + length]

                    steps = [WorkflowStep(tool_name=t) for t in sub_seq]
                    sig = WorkflowSignature(steps=steps)
                    key = sig.fingerprint

                    if key not in workflow_map:
                        workflow_map[key] = DetectedWorkflow(signature=sig)

                    workflow_map[key].occurrences.append(
                        WorkflowOccurrence(
                            trace_id=entry["trace_id"],
                            step_indices=(i, i + length - 1),
                            success=entry["success"],
                            parameters=sub_params,
                        )
                    )

        return workflow_map

    def _filter_and_merge(
        self, workflow_map: dict[str, DetectedWorkflow]
    ) -> list[DetectedWorkflow]:
        candidates = [
            wf
            for wf in workflow_map.values()
            if wf.occurrence_count >= self.min_occurrences
            and wf.success_rate >= self.min_success_rate
        ]
        candidates.sort(key=lambda wf: wf.confidence, reverse=True)
        return candidates


class WorkflowCompiler:
    def __init__(self):
        self._compiled_count = 0

    def compile(
        self,
        workflow: DetectedWorkflow,
        domain: str = "",
    ) -> Tool:
        steps = workflow.signature.steps
        tool_names = [s.tool_name for s in steps]

        name = self._generate_tool_name(tool_names, domain)
        param_schema = workflow.parameter_schema()
        description = self._generate_description(tool_names, workflow)

        fn = self._create_executor(tool_names, param_schema)

        return Tool(
            name=name,
            description=description,
            parameters=param_schema,
            fn=fn,
        )

    def _generate_tool_name(self, tool_names: list[str], domain: str) -> str:
        self._compiled_count += 1
        if len(tool_names) <= 2:
            base = "_".join(tool_names)
        else:
            base = f"{tool_names[0]}_thru_{tool_names[-1]}"

        unique = hashlib.md5("\x00".join(tool_names).encode()).hexdigest()[:8]
        return f"evolved_{base}_{unique}"

    def _generate_description(
        self, tool_names: list[str], workflow: DetectedWorkflow
    ) -> str:
        steps_desc = " \u2192 ".join(tool_names)
        return (
            f"Evolved workflow: {steps_desc}. "
            f"Success rate: {workflow.success_rate:.0%} over {workflow.occurrence_count} runs. "
            f"Executes the full {len(tool_names)}-step sequence in one call."
        )

    def _create_executor(
        self,
        tool_names: list[str],
        param_schema: dict[str, Any],
    ) -> Callable[..., Any]:
        def executor(**kwargs: Any) -> Any:
            results = {}
            last_output = None

            for step_idx, tool_name in enumerate(tool_names):
                step_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k in param_schema or step_idx == 0
                }
                if last_output is not None and isinstance(last_output, dict):
                    step_kwargs.update(last_output)

                from nil.tools.registry import get_tool

                tool = get_tool(tool_name)
                if tool is None:
                    raise RuntimeError(
                        f"Evolved tool '{' '.join(tool_names)}': "
                        f"underlying tool '{tool_name}' not found in registry"
                    )

                try:
                    output = tool.invoke(step_kwargs)
                    results[tool_name] = output
                    last_output = output
                except Exception as e:
                    raise RuntimeError(
                        f"Evolved tool failed at step {step_idx} ('{tool_name}'): {e}"
                    ) from e

            return {"steps": results, "final": last_output}

        executor.__name__ = f"evolved_{'_'.join(tool_names)}"
        return executor


@dataclass
class EvolutionResult:
    tools_evolved: list[Tool] = field(default_factory=list)
    workflows_detected: int = 0
    workflows_compiled: int = 0
    duration: float = 0.0
    error: Optional[str] = None


class ToolEvolver:
    def __init__(
        self,
        min_occurrences: int = 3,
        min_success_rate: float = 0.7,
        auto_register: bool = True,
    ):
        self.detector = WorkflowDetector(
            min_occurrences=min_occurrences,
            min_success_rate=min_success_rate,
        )
        self.compiler = WorkflowCompiler()
        self.auto_register = auto_register
        self._evolved_fingerprints: set[str] = set()

    def evolve(
        self,
        traces: list[Any],
        domain: str = "",
    ) -> EvolutionResult:
        start = time.time()
        result = EvolutionResult()

        try:
            workflows = self.detector.detect(traces)
            result.workflows_detected = len(workflows)

            if not workflows:
                result.duration = time.time() - start
                return result

            new_workflows = [
                wf
                for wf in workflows
                if wf.signature.fingerprint not in self._evolved_fingerprints
            ]

            if not new_workflows:
                result.duration = time.time() - start
                return result

            for workflow in new_workflows:
                try:
                    tool = self.compiler.compile(workflow, domain=domain)
                    self._evolved_fingerprints.add(workflow.signature.fingerprint)

                    if self.auto_register:
                        register_tool(tool)
                        logger.info(
                            "Evolved and registered tool '%s' from workflow: %s "
                            "(confidence=%.2f, %d occurrences)",
                            tool.name,
                            workflow.signature.normalized_key,
                            workflow.confidence,
                            workflow.occurrence_count,
                        )

                    result.tools_evolved.append(tool)
                    result.workflows_compiled += 1

                except Exception as e:
                    logger.warning(
                        "Failed to compile workflow '%s': %s",
                        workflow.signature.normalized_key,
                        e,
                    )

        except Exception as e:
            logger.error("Tool evolution failed: %s", e)
            result.error = str(e)

        result.duration = time.time() - start
        return result

    def get_evolved_tools(self) -> list[Tool]:
        from nil.tools.registry import get_tool, list_tools

        evolved = []
        for name in list_tools():
            if name.startswith("evolved_"):
                t = get_tool(name)
                if t:
                    evolved.append(t)
        return evolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "evolved_fingerprints": list(self._evolved_fingerprints),
            "auto_register": self.auto_register,
            "detector": {
                "min_occurrences": self.detector.min_occurrences,
                "min_success_rate": self.detector.min_success_rate,
            },
        }
