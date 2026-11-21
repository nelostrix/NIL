"""Lazy training APIs: load each feature only when it is requested."""

from importlib import import_module

_EXPORTS = {
    'checkpoint': ('nil.training.checkpoint', None),
    'EarlyStopping': ('nil.training.callbacks', 'EarlyStopping'),
    'SearchResult': ('nil.training.search', 'SearchResult'),
    'grid_search': ('nil.training.search', 'grid_search'),
    'CausalLM': ('nil.training.tasks', 'CausalLM'),
    'Supervised': ('nil.training.tasks', 'Supervised'),
    'Trainer': ('nil.training.trainer', 'Trainer'),
    'TrainingConfig': ('nil.training.trainer', 'TrainingConfig'),
    'TrainingResult': ('nil.training.trainer', 'TrainingResult'),
    'train': ('nil.training.trainer', 'train'),
    'BenchmarkResult': ('nil.training.curriculum', 'BenchmarkResult'),
    'BenchmarkSuite': ('nil.training.curriculum', 'BenchmarkSuite'),
    'CurriculumEngine': ('nil.training.curriculum', 'CurriculumEngine'),
    'PracticeEngine': ('nil.training.curriculum', 'PracticeEngine'),
    'PracticeSession': ('nil.training.curriculum', 'PracticeSession'),
    'SyntheticTaskGenerator': ('nil.training.curriculum', 'SyntheticTaskGenerator'),
    'WeaknessAnalyzer': ('nil.training.curriculum', 'WeaknessAnalyzer'),
    'WeaknessProfile': ('nil.training.curriculum', 'WeaknessProfile'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    loaded = import_module(module)
    value = loaded if symbol is None else getattr(loaded, symbol)
    globals()[name] = value
    return value
