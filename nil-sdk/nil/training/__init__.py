"""Model training, search, callbacks and autonomous curriculum utilities."""

from nil.training.callbacks import EarlyStopping
from nil.training.curriculum import (
    BenchmarkResult,
    BenchmarkSuite,
    CurriculumEngine,
    PracticeEngine,
    PracticeSession,
    SyntheticTaskGenerator,
    WeaknessAnalyzer,
    WeaknessProfile,
)
from nil.training.search import SearchResult, grid_search
from nil.training.tasks import CausalLM, Supervised
from nil.training.trainer import Trainer, TrainingConfig, TrainingResult, train

__all__ = [
    "EarlyStopping",
    "SearchResult",
    "grid_search",
    "Trainer",
    "TrainingConfig",
    "TrainingResult",
    "train",
    "Supervised",
    "CausalLM",
    "WeaknessAnalyzer",
    "WeaknessProfile",
    "SyntheticTaskGenerator",
    "PracticeEngine",
    "PracticeSession",
    "BenchmarkSuite",
    "BenchmarkResult",
    "CurriculumEngine",
]
