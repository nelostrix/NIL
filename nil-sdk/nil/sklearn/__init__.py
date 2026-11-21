"""Scikit-learn's public API through NIL, using an explicit optional backend.

Examples::

    from nil.sklearn.linear_model import LogisticRegression
    from nil.sklearn.model_selection import train_test_split
    from nil.sklearn.feature_extraction.text import TfidfVectorizer

No backend is loaded until an API/submodule is requested. Models and functions
are the original sklearn objects, not reimplementations or bundled copies.
"""

from nil._sklearn_mirror import install as _install, resolve_root as __getattr__

_install()
del _install


def __dir__():
    from nil.runtime import require
    backend = require('sklearn', 'classical')
    return sorted(set(globals()) | set(backend.__all__) | {'base', 'utils', '__version__'})
