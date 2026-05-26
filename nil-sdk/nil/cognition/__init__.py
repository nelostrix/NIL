"""Retrieval-Augmented Cognition — dynamic retrieval across skills, failures, trajectories, constraints, and facts."""

from nil.cognition.assembler import ContextAssembler
from nil.cognition.feedback import RetrievalFeedback
from nil.cognition.index import RetrievalIndex, ScoredResult, StoredItem
from nil.cognition.query import RetrievalQuery, RetrievalStrategy, RetrievalStrategySelector
from nil.cognition.retrieval import HybridRetriever
from nil.cognition.runner import RACEngine
from nil.cognition.scoring import RelevanceScorer
from nil.cognition.sources import MultiSourceRetriever

__all__ = [
    "RetrievalQuery",
    "RetrievalStrategy",
    "RetrievalStrategySelector",
    "StoredItem",
    "ScoredResult",
    "RetrievalIndex",
    "HybridRetriever",
    "RelevanceScorer",
    "ContextAssembler",
    "RetrievalFeedback",
    "MultiSourceRetriever",
    "RACEngine",
]
