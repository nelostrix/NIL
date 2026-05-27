"""Retrieval-Augmented Cognition — dynamic retrieval across skills, failures, trajectories, constraints, and facts.

Also includes the Unified Cognitive Loop — a 10-phase observe→retrieve→reason→plan→simulate→act→evaluate→reflect→learn→adapt
pipeline that ALL subsystems plug into, plus:
  - Executive Cognition Layer (CognitiveMode, CognitivePolicyEngine, recursive cognition)
  - Internal Simulation (candidate generation, rollouts, reward/cost/uncertainty estimation)
  - Live Cognition / Online Adaptation (mid-task supervision with triggers and actions)
"""

from nil.cognition.adapters import AgentSpawner, ModelSwitcher, ReasoningDeeper, RetrievalAlterer
from nil.cognition.assembler import ContextAssembler
from nil.cognition.feedback import RetrievalFeedback
from nil.cognition.index import RetrievalIndex, ScoredResult, StoredItem
from nil.cognition.loop import CognitiveLoop, CognitiveLoopPlugin, LoopResult, Phase, PhaseContext
from nil.cognition.modes import CognitiveMode, ModeProfile, MODE_PROFILES, Uncertainty
from nil.cognition.plugins import (
    EvolutionPlugin,
    KernelPlugin,
    MemoryPlugin,
    RACPlugin,
    RLPlugin,
    wire_cognitive_loop,
)
from nil.cognition.policy import CognitivePolicyEngine, build_default_policy
from nil.cognition.query import RetrievalQuery, RetrievalStrategy, RetrievalStrategySelector
from nil.cognition.retrieval import HybridRetriever
from nil.cognition.runner import RACEngine
from nil.cognition.scoring import RelevanceScorer
from nil.cognition.simulation import SimulationCandidate, SimulationEngine
from nil.cognition.sources import MultiSourceRetriever
from nil.cognition.supervisor import LiveSupervisor, SupervisionReport

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
    # Unified Cognitive Loop
    "CognitiveLoop",
    "CognitiveLoopPlugin",
    "LoopResult",
    "Phase",
    "PhaseContext",
    # Executive Cognition Layer
    "CognitiveMode",
    "ModeProfile",
    "MODE_PROFILES",
    "Uncertainty",
    "CognitivePolicyEngine",
    "build_default_policy",
    # Internal Simulation
    "SimulationCandidate",
    "SimulationEngine",
    # Live Cognition / Online Adaptation
    "LiveSupervisor",
    "SupervisionReport",
    "ModelSwitcher",
    "AgentSpawner",
    "RetrievalAlterer",
    "ReasoningDeeper",
    # Plugins
    "MemoryPlugin",
    "RACPlugin",
    "KernelPlugin",
    "EvolutionPlugin",
    "RLPlugin",
    "wire_cognitive_loop",
]
