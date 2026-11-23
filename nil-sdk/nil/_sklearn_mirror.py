"""Scoped import support for public sklearn APIs under nil.sklearn.

Classes/functions remain native objects; only namespace modules are proxies.
The finder is installed only when nil.sklearn is explicitly imported.
"""

from importlib import import_module, util
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
import sys

from nil.runtime import require

_PREFIX = 'nil.sklearn.'


def _public(path):
    return all(part.isidentifier() and not part.startswith('_') for part in path.split('.'))


class _MirrorLoader(Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        backend = import_module(self.target)
        module.__doc__ = backend.__doc__
        module.__all__ = [name for name in getattr(backend, '__all__', dir(backend))
                          if not name.startswith('_')]

        def resolve(name):
            # Modules retain separate import metadata. Objects obtained from them
            # are native, preserving clone, pickle and isinstance behavior.
            if name.startswith('_'):
                raise AttributeError(f'module {module.__name__!r} has no attribute {name!r}')
            return getattr(backend, name)

        module.__getattr__ = resolve
        module.__dir__ = lambda: sorted(set(module.__dict__) | set(dir(backend)))


class _MirrorFinder(MetaPathFinder):
    _nil_sklearn_mirror = True

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        suffix = fullname[len(_PREFIX):]
        if not _public(suffix):
            return None
        require('sklearn', 'classical')
        native = 'sklearn.' + suffix
        try:
            spec = util.find_spec(native)
        except ModuleNotFoundError as error:
            if error.name == native or native.startswith(f'{error.name}.'):
                return None
            raise
        if spec is None:
            return None
        return ModuleSpec(fullname, _MirrorLoader(native),
                          is_package=spec.submodule_search_locations is not None)


def install():
    if not any(getattr(finder, '_nil_sklearn_mirror', False) for finder in sys.meta_path):
        sys.meta_path.append(_MirrorFinder())


def resolve_root(name):
    if name.startswith('_') and name not in {'__version__', '__all__'}:
        raise AttributeError(f'module nil.sklearn has no attribute {name!r}')
    backend = require('sklearn', 'classical')
    if name in {'__version__', '__all__'}:
        return getattr(backend, name)
    value = getattr(backend, name, None)
    if value is not None and not hasattr(value, '__path__'):
        return value
    if not _public(name):
        raise AttributeError(f'module nil.sklearn has no attribute {name!r}')
    try:
        return import_module(_PREFIX + name)
    except ModuleNotFoundError as error:
        if error.name == _PREFIX + name:
            raise AttributeError(f'module nil.sklearn has no attribute {name!r}') from error
        raise
