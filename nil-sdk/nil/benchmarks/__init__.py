"""Long-horizon benchmarking — measure learning, adaptation, memory, and robustness across runs."""

from nil.benchmarks.horizon import LongHorizonBenchmark
from nil.benchmarks.learning import (
    ContextDependenceBenchmark,
    LearningCurveBenchmarks,
    LearningVelocity,
    SequentialReasoningBenchmark,
    ToolSelectionBenchmark,
)
from nil.benchmarks.memory import MemoryRetentionBenchmark
from nil.benchmarks.recovery import RecoveryBenchmark
from nil.benchmarks.runner import BenchmarkReport, CompositeScore, NilBenchmarkSuite
from nil.benchmarks.system import SystemBenchmark
from nil.benchmarks.transfer import TransferBenchmark

__all__ = [
    "LearningVelocity",
    "ToolSelectionBenchmark",
    "SequentialReasoningBenchmark",
    "ContextDependenceBenchmark",
    "LearningCurveBenchmarks",
    "LongHorizonBenchmark",
    "RecoveryBenchmark",
    "MemoryRetentionBenchmark",
    "TransferBenchmark",
    "SystemBenchmark",
    "NilBenchmarkSuite",
    "BenchmarkReport",
    "CompositeScore",
]
