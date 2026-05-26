from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from nil.observability.events import ObservabilityEvent, ObservabilityStore


@dataclass
class EvolutionDiff:
    evolution_id: str = ""
    trigger: str = ""
    duration: float = 0.0
    traces_analyzed: int = 0
    analysis: str = ""
    extraction: str = ""
    gradients: str = ""
    evaluation: str = ""
    world_model: str = ""
    drift: str = ""
    strategy_diff: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "evolution_id": self.evolution_id,
            "trigger": self.trigger,
            "duration": self.duration,
            "traces_analyzed": self.traces_analyzed,
            "analysis": self.analysis,
            "extraction": self.extraction,
            "gradients": self.gradients,
            "evaluation": self.evaluation,
            "world_model": self.world_model,
            "drift": self.drift,
            "strategy_diff": self.strategy_diff,
        }


class EvolutionDiffViewer:
    def __init__(self, store: ObservabilityStore):
        self.store = store

    def render(self, evolution_id: str) -> EvolutionDiff:
        events = self.store.get_events(
            event_type=None, run_id=evolution_id
        )

        pipeline_result = self._extract_pipeline_result(events)

        return EvolutionDiff(
            evolution_id=evolution_id,
            trigger=pipeline_result.get("trigger", "unknown"),
            duration=pipeline_result.get("duration", 0),
            traces_analyzed=pipeline_result.get("traces_analyzed", 0),
            analysis=self._render_analysis_stage(events),
            extraction=self._render_extraction_stage(events),
            gradients=self._render_gradients_stage(events),
            evaluation=self._render_evaluation_stage(events),
            world_model=self._render_world_model_stage(events),
            drift=self._render_drift_stage(events),
            strategy_diff=self._compute_strategy_diff(
                pipeline_result.get("strategy_before", {}),
                pipeline_result.get("strategy_after", {}),
            ),
        )

    def render_text(self, evolution_id: str) -> str:
        diff = self.render(evolution_id)
        lines = [
            f"Evolution Run #{evolution_id}",
            f"  Trigger: {diff.trigger}",
            f"  Duration: {diff.duration}s",
            f"  Traces analyzed: {diff.traces_analyzed}",
            "",
            f"  Analysis: {diff.analysis or 'N/A'}",
            f"  Extraction: {diff.extraction or 'N/A'}",
            f"  Gradients: {diff.gradients or 'N/A'}",
            f"  Evaluation: {diff.evaluation or 'N/A'}",
            f"  World Model: {diff.world_model or 'N/A'}",
            f"  Drift: {diff.drift or 'N/A'}",
            "",
            f"  Strategy diff:",
            f"    {diff.strategy_diff or 'No changes'}",
        ]
        return "\n".join(lines)

    def _extract_pipeline_result(self, events: list[ObservabilityEvent]) -> dict[str, Any]:
        for e in events:
            if e.event_type.value == "evolution_triggered":
                return e.payload
        return {}

    def _render_analysis_stage(self, events: list[ObservabilityEvent]) -> str:
        for e in events:
            if e.event_type.value == "analysis_completed":
                n_patterns = e.payload.get("n_patterns", 0)
                n_significant = e.payload.get("n_significant", 0)
                return f"{n_patterns} causal patterns found, {n_significant} significant"
        return ""

    def _render_extraction_stage(self, events: list[ObservabilityEvent]) -> str:
        for e in events:
            if e.event_type.value == "skill_extracted":
                name = e.payload.get("skill_name", "unknown")
                conf = e.payload.get("confidence", 0)
                return f"1 new skill extracted: {name} ({conf:.0%})"
        return ""

    def _render_gradients_stage(self, events: list[ObservabilityEvent]) -> str:
        for e in events:
            if e.event_type.value == "gradient_computed":
                n = e.payload.get("n_critiques", 0)
                m = e.payload.get("n_deltas", 0)
                return f"{n} critiques → {m} deltas"
        return ""

    def _render_evaluation_stage(self, events: list[ObservabilityEvent]) -> str:
        for e in events:
            if e.event_type.value == "delta_applied":
                n = e.payload.get("n_promoted", 0)
                return f"{n} deltas promoted"
        return ""

    def _render_world_model_stage(self, events: list[ObservabilityEvent]) -> str:
        return "rebuilt"

    def _render_drift_stage(self, events: list[ObservabilityEvent]) -> str:
        for e in events:
            if e.event_type.value == "drift_detected":
                sev = e.payload.get("severity", "none")
                return f"{sev} drift detected"
        return "no drift detected"

    def _compute_strategy_diff(self, before: dict, after: dict) -> str:
        before_keys = set(before.keys())
        after_keys = set(after.keys())
        added = after_keys - before_keys
        removed = before_keys - after_keys
        parts = []
        if added:
            parts.append(f"+{', '.join(added)}")
        if removed:
            parts.append(f"-{', '.join(removed)}")
        return "; ".join(parts) if parts else "No structural changes"
