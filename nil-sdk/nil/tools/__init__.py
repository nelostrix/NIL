"""Tools package - tool definations, registry, embeddings, affordances, and latent selection."""

from nil.tools.base import Tool, tool
from nil.tools.embeddings import ToolEmbedding, ToolEmbeddingGenerator
from nil.tools.affordance import ToolAffordance, AffordanceLearner, ToolAffordanceGraph
from nil.tools.composition import CompositeTool, ToolCompositionSynthesizer
from nil.tools.selection import LatentToolSelector, LatentAwarePromptBuilder, ScoredTool

__all__ = [
    "tool", "Tool",
    "ToolEmbedding", "ToolEmbeddingGenerator",
    "ToolAffordance", "AffordanceLearner", "ToolAffordanceGraph",
    "CompositeTool", "ToolCompositionSynthesizer",
    "LatentToolSelector", "LatentAwarePromptBuilder", "ScoredTool",
]
