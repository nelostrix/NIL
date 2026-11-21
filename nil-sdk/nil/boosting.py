"""Explicit optional native boosting constructors; no backend imports until called."""
from nil.runtime import require


def lightgbm_classifier(**parameters):
    return require('lightgbm', 'lightgbm').LGBMClassifier(**{'n_jobs': 1, 'verbosity': -1, **parameters})


def lightgbm_regressor(**parameters):
    return require('lightgbm', 'lightgbm').LGBMRegressor(**{'n_jobs': 1, 'verbosity': -1, **parameters})


def catboost_classifier(**parameters):
    return require('catboost', 'catboost').CatBoostClassifier(**{'iterations': 100, 'thread_count': 1, 'task_type': 'CPU', 'verbose': False, 'allow_writing_files': False, **parameters})


def catboost_regressor(**parameters):
    return require('catboost', 'catboost').CatBoostRegressor(**{'iterations': 100, 'thread_count': 1, 'task_type': 'CPU', 'verbose': False, 'allow_writing_files': False, **parameters})
