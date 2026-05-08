"""
Evolution Pipeline — wires all 6 modules into a single self-improvement loop.

This is the orchestrator that turns individual modules into a closed loop:

    traces → analyzer → gradient engine → skill extractor
           → world model → evaluator → drift detector
           → better agent → better traces → ...

The pipeline runs in stages. Not every stage runs every time:

    STAGE 1: ANALYZE (always runs)
        Run causal analysis on accumulated traces.
        Fast, no LLM calls, pure pattern mining.

    STAGE 2: EXTRACT SKILLS (runs if new patterns found)
        Compress winning patterns into reusable skills.
        Store in memory for prompt injection.

    STAGE 3: COMPUTE GRADIENTS (runs if enough failures)
        Critique failed traces, propose strategy improvements.
        Uses LLM if available, falls back to heuristic stub.

    STAGE 4: EVALUATE CANDIDATES (runs if gradients produced deltas)
        Replay-test candidate strategies against historical traces.
        Only promote if statistically significant improvement.

    STAGE 5: UPDATE WORLD MODEL (runs periodically)
        Rebuild the domain's internal knowledge representation.
        Inject into system prompt for next runs.

    STAGE 6: CHECK DRIFT (always runs)
        Monitor for performance degradation.
        Trigger auto-recovery if critical drift detected.

Usage:
    pipeline = EvolutionPipeline(memory)
    result = pipeline.run("my-domain")

    # Or with custom config:
    pipeline = EvolutionPipeline(
        memory,
        gradient_provider=AnthropicGradientProvider(),
        auto_apply=True,
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.evolution.analyzer import AnalysisReport, TraceAnalyzer
from nil.evolution.drift_detector import DriftDetector, DriftReport, RecoveryResult
from nil.evolution.evaluator import EvaluationResult, ReplayEvaluator
from nil.evolution.gradient_engine import (
    GradientBatch,
    GradientProvider,
    StrategyDelta,
    TextualGradientEngine,
)
from nil.evolution.skill_extractor import ExtractionResult, SkillExtractor
from nil.evolution.world_model import WorldModel, WorldModelBuilder
from nil.memory.store import PersistentMemory

logger = logging.getLogger("nil.evolution.pipeline")


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Complete result of one evolution pipeline run."""

    domain: str
    stages_run: list[str] = field(default_factory=list)
    analysis: Optional[AnalysisReport] = None
    extraction: Optional[ExtractionResult] = None
    gradient_batch: Optional[GradientBatch] = None
    evaluations: list[EvaluationResult] = field(default_factory=list)
    world_model: Optional[WorldModel] = None
    drift_report: Optional[DriftReport] = None
    recovery: Optional[RecoveryResult] = None

    # What actually changed
    skills_extracted: int = 0
    skills_deprecated: int = 0
    deltas_applied: int = 0
    strategy_promoted: bool = False
    strategy_version: int = 0
    drift_detected: bool = False
    recovery_triggered: bool = False

    duration: float = 0.0
    run_at: float = field(default_factory=time.time)
    error: Optional[str] = None

    @property
    def had_impact(self) -> bool:
        """Whether this pipeline run changed anything."""
        return (
            self.skills_extracted > 0
            or self.deltas_applied > 0
            or self.strategy_promoted
            or self.recovery_triggered
        )

    def summary(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "stages": self.stages_run,
            "skills_extracted": self.skills_extracted,
            "skills_deprecated": self.skills_deprecated,
            "deltas_applied": self.deltas_applied,
            "strategy_promoted": self.strategy_promoted,
            "strategy_version": self.strategy_version,
            "drift_detected": self.drift_detected,
            "recovery_triggered": self.recovery_triggered,
            "had_impact": self.had_impact,
            "duration_s": round(self.duration, 2),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class EvolutionPipeline:
    """
    Orchestrates the full evolution loop.

    This is the single entry point that runs all 6 modules
    in sequence, making decisions about what to skip and
    what to apply.

    Usage:
        pipeline = EvolutionPipeline(memory)
        result = pipeline.run("my-domain")

        if result.had_impact:
            print(f"Evolution improved the agent: {result.summary()}")
    """

    def __init__(
        self,
        memory: PersistentMemory,
        analyzer: Optional[TraceAnalyzer] = None,
        gradient_provider: Optional[GradientProvider] = None,
        auto_apply: bool = True,
        min_traces_for_evolution: int = 10,
        min_traces_for_world_model: int = 20,
        world_model_rebuild_interval: int = 50,
    ) -> None:
        self.memory = memory

        # Shared analyzer instance
        self.analyzer = analyzer or TraceAnalyzer(memory)

        # Module instances (share analyzer for consistency)
        self.gradient_engine = TextualGradientEngine(
            memory,
            provider=gradient_provider,
            analyzer=self.analyzer,
        )
        self.skill_extractor = SkillExtractor(memory, analyzer=self.analyzer)
        self.world_model_builder = WorldModelBuilder(memory, analyzer=self.analyzer)
        self.evaluator = ReplayEvaluator(memory, analyzer=self.analyzer)
        self.drift_detector = DriftDetector(memory, analyzer=self.analyzer)

        self.auto_apply = auto_apply
        self.min_traces = min_traces_for_evolution
        self.min_traces_world = min_traces_for_world_model
        self.world_model_interval = world_model_rebuild_interval

        # Track state across pipeline runs
        self._run_count = 0
        self._last_world_model_at = 0  # trace count when last built

    def run(
        self,
        domain: str,
        max_traces: int = 300,
    ) -> PipelineResult:
        """
        Execute the full evolution pipeline.

        Runs each stage in order, skipping stages that don't
        have enough data or aren't needed.
        """
        start = time.time()
        self._run_count += 1
        result = PipelineResult(domain=domain)

        trace_count = self.memory.get_trace_count(domain)

        if trace_count < self.min_traces:
            logger.info(
                "Pipeline skip for '%s': %d traces (need %d)",
                domain,
                trace_count,
                self.min_traces,
            )
            result.stages_run.append("skipped:insufficient_data")
            result.duration = time.time() - start
            return result

        logger.info(
            "Evolution pipeline starting for '%s' (%d traces, run #%d)",
            domain,
            trace_count,
            self._run_count,
        )

        try:
            # STAGE 1: Analyze
            result.analysis = self._stage_analyze(domain, max_traces)
            result.stages_run.append("analyze")

            # STAGE 2: Extract skills
            result.extraction = self._stage_extract_skills(domain, max_traces)
            result.stages_run.append("extract_skills")
            if result.extraction:
                result.skills_extracted = result.extraction.new_skill_count
                result.skills_deprecated = len(result.extraction.deprecated)

            # STAGE 3: Compute gradients
            result.gradient_batch = self._stage_compute_gradients(domain, max_traces)
            result.stages_run.append("compute_gradients")

            # STAGE 4: Evaluate and apply
            if result.gradient_batch and result.gradient_batch.has_actionable_deltas:
                evals, applied, promoted, version = self._stage_evaluate_and_apply(
                    domain, result.gradient_batch, max_traces
                )
                result.evaluations = evals
                result.deltas_applied = applied
                result.strategy_promoted = promoted
                result.strategy_version = version
                result.stages_run.append("evaluate_and_apply")

            # STAGE 5: Update world model (periodically)
            if self._should_rebuild_world_model(trace_count):
                result.world_model = self._stage_build_world_model(domain, max_traces)
                result.stages_run.append("build_world_model")
                self._last_world_model_at = trace_count

            # STAGE 6: Check drift
            result.drift_report = self._stage_check_drift(domain, max_traces)
            result.stages_run.append("check_drift")
            if result.drift_report:
                result.drift_detected = result.drift_report.drift_detected
                if result.drift_detected and self.auto_apply:
                    result.recovery = self._stage_auto_recover(domain)
                    result.recovery_triggered = (
                        result.recovery is not None and result.recovery.recovery_triggered
                    )
                    if result.recovery_triggered:
                        result.stages_run.append("auto_recover")

        except Exception as e:
            logger.error("Pipeline error for '%s': %s", domain, str(e))
            result.error = str(e)
            result.stages_run.append(f"error:{type(e).__name__}")

        result.duration = time.time() - start

        logger.info(
            "Pipeline complete for '%s' in %.2fs: stages=%s, impact=%s",
            domain,
            result.duration,
            ",".join(result.stages_run),
            result.had_impact,
        )

        return result

    # ------------------------------------------------------------------
    # Individual stages
    # ------------------------------------------------------------------

    def _stage_analyze(self, domain: str, max_traces: int) -> AnalysisReport:
        """Stage 1: Run causal analysis."""
        logger.debug("Stage 1: Analyzing traces for '%s'", domain)
        return self.analyzer.full_analysis(domain, max_traces=max_traces)

    def _stage_extract_skills(self, domain: str, max_traces: int) -> ExtractionResult:
        """Stage 2: Extract and compress skills."""
        logger.debug("Stage 2: Extracting skills for '%s'", domain)
        return self.skill_extractor.extract(domain, max_traces=max_traces)

    def _stage_compute_gradients(self, domain: str, max_traces: int) -> GradientBatch:
        """Stage 3: Generate textual gradients."""
        logger.debug("Stage 3: Computing gradients for '%s'", domain)
        return self.gradient_engine.compute_gradients(domain, max_traces=max_traces)

    def _stage_evaluate_and_apply(
        self,
        domain: str,
        batch: GradientBatch,
        max_traces: int,
    ) -> tuple[list[EvaluationResult], int, bool, int]:
        """
        Stage 4: Evaluate candidate deltas and apply the best ones.

        Returns: (evaluations, deltas_applied, strategy_promoted, version)
        """
        logger.debug("Stage 4: Evaluating %d deltas for '%s'", len(batch.deltas), domain)

        if not batch.deltas:
            return [], 0, False, 0

        # Convert StrategyDelta objects to dicts for evaluator
        delta_dicts = [{"delta_id": d.delta_id, "changes": d.changes} for d in batch.deltas]

        # Evaluate all deltas
        evaluations = self.evaluator.evaluate_deltas(domain, delta_dicts, max_traces=max_traces)

        if not self.auto_apply:
            return evaluations, 0, False, 0

        # Find promotable deltas
        promotable = [
            (eval_result, batch.deltas[i] if i < len(batch.deltas) else None)
            for i, eval_result in enumerate(evaluations)
            if eval_result.recommendation == "promote"
        ]

        if not promotable:
            # If nothing is statistically significant, apply high-confidence
            # analysis-driven deltas anyway (they're based on data, not LLM)
            analysis_deltas = [
                d
                for d in batch.deltas
                if d.source_gradient == "analysis" and d.expected_impact >= 0.4
            ]
            if analysis_deltas:
                logger.info(
                    "No statistically significant deltas, but applying %d "
                    "analysis-driven deltas for '%s'",
                    len(analysis_deltas),
                    domain,
                )
                self.gradient_engine.apply_deltas(domain, analysis_deltas)
                strategy = self.memory.get_active_strategy(domain)
                version = 0
                if strategy:
                    # Get version from the last stored strategy
                    conn = self.memory._get_conn()
                    row = conn.execute(
                        "SELECT MAX(version) as v FROM strategy_versions WHERE domain=?",
                        (domain,),
                    ).fetchone()
                    version = row["v"] if row and row["v"] else 0

                return evaluations, len(analysis_deltas), False, version

            return evaluations, 0, False, 0

        # Apply the best promotable deltas
        deltas_to_apply = [d for _, d in promotable if d is not None]
        if deltas_to_apply:
            self.gradient_engine.apply_deltas(domain, deltas_to_apply)

            # Get new version number
            conn = self.memory._get_conn()
            row = conn.execute(
                "SELECT MAX(version) as v FROM strategy_versions WHERE domain=?",
                (domain,),
            ).fetchone()
            version = row["v"] if row and row["v"] else 0

            logger.info(
                "Promoted %d deltas for '%s' → strategy v%d",
                len(deltas_to_apply),
                domain,
                version,
            )
            return evaluations, len(deltas_to_apply), True, version

        return evaluations, 0, False, 0

    def _stage_build_world_model(self, domain: str, max_traces: int) -> WorldModel:
        """Stage 5: Build/rebuild the domain world model."""
        logger.debug("Stage 5: Building world model for '%s'", domain)
        return self.world_model_builder.build(domain, max_traces=max_traces)

    def _stage_check_drift(self, domain: str, max_traces: int) -> DriftReport:
        """Stage 6: Check for performance drift."""
        logger.debug("Stage 6: Checking drift for '%s'", domain)
        return self.drift_detector.check(domain, max_traces=max_traces)

    def _stage_auto_recover(self, domain: str) -> RecoveryResult:
        """Stage 6b: Auto-recover from drift."""
        logger.info("Auto-recovery triggered for '%s'", domain)
        _, recovery = self.drift_detector.check_and_recover(domain)
        return recovery

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_rebuild_world_model(self, current_trace_count: int) -> bool:
        """Decide if the world model needs rebuilding."""
        if current_trace_count < self.min_traces_world:
            return False
        if self._last_world_model_at == 0:
            return True  # Never built
        return (current_trace_count - self._last_world_model_at) >= self.world_model_interval

    def get_world_model(self, domain: str) -> WorldModel:
        """Build and return the current world model (on demand)."""
        return self.world_model_builder.build(domain)

    def get_skill_prompt(self, domain: str, compact: bool = False) -> str:
        """Get the current skill injection prompt for a domain."""
        return self.skill_extractor.generate_skill_prompt(domain, compact=compact)
