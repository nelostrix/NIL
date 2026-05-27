"""Autonomous Curriculum — self-directed capability growth through weakness detection and targeted practice."""

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

__all__ = [
    "WeaknessAnalyzer",
    "WeaknessProfile",
    "SyntheticTaskGenerator",
    "PracticeEngine",
    "PracticeSession",
    "BenchmarkSuite",
    "BenchmarkResult",
    "CurriculumEngine",
]
