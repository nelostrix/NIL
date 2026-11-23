"""Real backend compatibility: public symbols, native types and NIL workflows."""

import importlib
import pickle

import pytest

sklearn = pytest.importorskip('sklearn')
import numpy as np

import nil


@pytest.mark.parametrize('name', sorted(set(sklearn.__all__) - {
    'clone', 'get_config', 'set_config', 'config_context', 'show_versions',
} | {'base', 'utils'}))
def test_public_modules_and_exports_are_native(name):
    native = importlib.import_module('sklearn.' + name)
    native_spec = native.__spec__
    mirror = importlib.import_module('nil.sklearn.' + name)
    assert mirror.__name__ == 'nil.sklearn.' + name
    assert native.__spec__ is native_spec
    exports = getattr(native, '__all__', [key for key in dir(native) if not key.startswith('_')])
    for symbol in exports:
        if symbol.startswith('_'):
            continue
        try:
            original = getattr(native, symbol)
        except ImportError as error:
            # Experimental APIs must retain their explicit enable-import gate.
            with pytest.raises(type(error)) as mirrored:
                getattr(mirror, symbol)
            assert str(mirrored.value) == str(error)
        else:
            assert getattr(mirror, symbol) is original, (name, symbol)


@pytest.mark.parametrize('path,symbol', [
    ('feature_extraction.text', 'TfidfVectorizer'),
    ('metrics.pairwise', 'cosine_similarity'),
    ('gaussian_process.kernels', 'RBF'),
    ('utils.validation', 'check_array'),
])
def test_nested_imports(path, symbol):
    native = importlib.import_module('sklearn.' + path)
    mirror = importlib.import_module('nil.sklearn.' + path)
    assert getattr(mirror, symbol) is getattr(native, symbol)


def test_root_helpers_version_and_missing_attributes():
    for name in ('clone', 'get_config', 'set_config', 'config_context', 'show_versions', '__version__'):
        assert getattr(nil.sklearn, name) is getattr(sklearn, name)
    assert nil.sklearn.__all__ == sklearn.__all__
    assert {'linear_model', 'base', 'utils', '__version__'} <= set(dir(nil.sklearn))
    with pytest.raises(AttributeError):
        getattr(nil.sklearn, 'imaginary_module')
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('nil.sklearn._config')
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('nil.sklearn.linear_model.imaginary_module')


def test_normal_python_import_syntax_and_native_serialization():
    from nil.sklearn import linear_model, clone
    from nil.sklearn.linear_model import LinearRegression
    from nil.sklearn.metrics import mean_squared_error

    assert LinearRegression is sklearn.linear_model.LinearRegression
    model = linear_model.LinearRegression().fit([[0], [1], [2]], [1, 3, 5])
    assert clone(model).get_params() == model.get_params()
    restored = pickle.loads(pickle.dumps(model))
    assert type(restored) is LinearRegression
    assert mean_squared_error([7], restored.predict([[3]])) == pytest.approx(0)


def test_text_pipeline_search_and_complete_nil_roundtrip(tmp_path):
    from nil.sklearn.feature_extraction.text import TfidfVectorizer
    from nil.sklearn.linear_model import LogisticRegression
    from nil.sklearn.pipeline import make_pipeline
    from nil.sklearn.model_selection import GridSearchCV

    texts = ['good happy', 'great happy', 'good bright', 'great bright',
             'bad sad', 'awful sad', 'bad dark', 'awful dark']
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(random_state=7))
    search = GridSearchCV(pipeline, {'logisticregression__C': [0.5, 1]}, cv=2, n_jobs=1)
    search.fit(texts, labels)
    result = nil.fit(search.best_estimator_, texts, labels,
                     validation=(['good happy', 'bad sad'], [1, 0]), output_dir=tmp_path)
    assert result.metrics['accuracy'] == 1
    result.save(tmp_path / 'text-model')
    loaded = nil.load(tmp_path / 'text-model', trusted=True)
    np.testing.assert_array_equal(loaded.predict(['great bright', 'awful dark']), [1, 0])


def test_incremental_and_multioutput_apis():
    from nil.sklearn.linear_model import SGDClassifier, Ridge
    from nil.sklearn.multioutput import MultiOutputRegressor
    from nil.sklearn.metrics import mean_squared_error

    classifier = SGDClassifier(loss='log_loss', random_state=7)
    classifier.partial_fit([[-2], [-1], [1], [2]], [0, 0, 1, 1], classes=[0, 1])
    classifier.partial_fit([[-3], [3]], [0, 1])
    np.testing.assert_array_equal(classifier.predict([[-5], [5]]), [0, 1])
    X = np.arange(20).reshape(-1, 1)
    y = np.column_stack((X[:, 0], 2 * X[:, 0]))
    model = MultiOutputRegressor(Ridge(alpha=0)).fit(X, y)
    assert mean_squared_error(y, model.predict(X)) == pytest.approx(0, abs=1e-20)


def test_experimental_enable_import_works_through_nil():
    from nil.sklearn.experimental import enable_halving_search_cv
    from nil.sklearn.model_selection import HalvingGridSearchCV

    assert enable_halving_search_cv is not None
    assert HalvingGridSearchCV is sklearn.model_selection.HalvingGridSearchCV
