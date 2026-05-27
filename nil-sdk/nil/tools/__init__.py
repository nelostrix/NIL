"""Tools package - tool definations, registry, embeddings, affordances, composition, evolution, and latent selection."""

from nil.tools.base import Tool, tool
from nil.tools.embeddings import ToolEmbedding, ToolEmbeddingGenerator
from nil.tools.affordance import ToolAffordance, AffordanceLearner, ToolAffordanceGraph
from nil.tools.composition import CompositeTool, ToolCompositionSynthesizer
from nil.tools.evolver import (
    DetectedWorkflow,
    EvolutionResult,
    ToolEvolver,
    WorkflowCompiler,
    WorkflowDetector,
    WorkflowOccurrence,
    WorkflowSignature,
    WorkflowStep,
)
from nil.tools.selection import LatentToolSelector, LatentAwarePromptBuilder, ScoredTool

__all__ = [
    "tool", "Tool",
    "ToolEmbedding", "ToolEmbeddingGenerator",
    "ToolAffordance", "AffordanceLearner", "ToolAffordanceGraph",
    "CompositeTool", "ToolCompositionSynthesizer",
    "LatentToolSelector", "LatentAwarePromptBuilder", "ScoredTool",
    # Tool Evolution (Phase 5)
    "WorkflowStep", "WorkflowSignature", "WorkflowOccurrence", "DetectedWorkflow",
    "WorkflowDetector", "WorkflowCompiler", "ToolEvolver", "EvolutionResult",
]
