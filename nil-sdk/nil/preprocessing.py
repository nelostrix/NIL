"""Native preprocessing pipelines fitted with the model, never on held-out data."""
from nil.runtime import require


def scaler(kind="standard"):
    """Choose a native scaler; standard/robust defaults preserve sparse matrices."""
    preprocessing = require("sklearn.preprocessing", "classical")
    if kind == "standard":
        return preprocessing.StandardScaler(with_mean=False)
    if kind == "robust":
        return preprocessing.RobustScaler(with_centering=False)
    if kind == "minmax":
        return preprocessing.MinMaxScaler()
    if kind == "maxabs":
        return preprocessing.MaxAbsScaler()
    raise ValueError("Choose standard, robust, minmax or maxabs scaling")


def select_features(*, k=10, task="classification"):
    if task not in {"classification", "regression"}:
        raise ValueError("Choose classification or regression for supervised feature selection")
    selection = require("sklearn.feature_selection", "classical")
    score = selection.f_classif if task == "classification" else selection.f_regression
    return selection.SelectKBest(score_func=score, k=k)


def numeric(*, impute="median", scale=True):
    Pipeline = require("sklearn.pipeline", "classical").Pipeline
    imputer = require("sklearn.impute", "classical").SimpleImputer
    steps = [("impute", imputer(strategy=impute, keep_empty_features=True))]
    if scale:
        steps.append(("scale", scaler("standard" if scale is True else scale)))
    return Pipeline(steps)


def categorical(*, impute="most_frequent", sparse=True):
    Pipeline = require("sklearn.pipeline", "classical").Pipeline
    imputer = require("sklearn.impute", "classical").SimpleImputer
    encoder = require("sklearn.preprocessing", "classical").OneHotEncoder
    return Pipeline([
        ("impute", imputer(strategy=impute, keep_empty_features=True)),
        ("encode", encoder(handle_unknown="ignore", sparse_output=sparse)),
    ])


def tabular(*, numeric_columns=(), categorical_columns=(), scale=True, sparse=True):
    """Select named DataFrame or integer array columns. Unselected columns are dropped.

    Missing values use numpy.nan; normalize other sentinels explicitly. Dense
    output can consume substantial memory. Sparse scaling does not center data.
    """
    if isinstance(numeric_columns, str) or isinstance(categorical_columns, str):
        raise TypeError("Pass column names as sequences, for example ['age']")
    numeric_columns, categorical_columns = list(numeric_columns), list(categorical_columns)
    selected = numeric_columns + categorical_columns
    if not selected:
        raise ValueError("Select at least one numeric or categorical column")
    if len(set(selected)) != len(selected):
        raise ValueError("A column may appear only once across selections")
    ColumnTransformer = require("sklearn.compose", "classical").ColumnTransformer
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric(scale=scale), numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical(sparse=sparse), categorical_columns))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0 if sparse else 0.0)


def pipeline(model, *, preprocess=None, select=None):
    from nil._classical_common import unwrap
    Pipeline = require("sklearn.pipeline", "classical").Pipeline
    steps = []
    if preprocess is not None:
        steps.append(("preprocess", preprocess))
    if select is not None:
        steps.append(("select", select))
    steps.append(("model", unwrap(model)))
    return Pipeline(steps)


def infer_columns(frame):
    """Infer supported DataFrame roles from dtypes, without fitting on values."""
    pd = require("pandas", "tabular")
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Automatic column inference requires a DataFrame with unique column names")
    roles = {"numeric": [], "categorical": []}
    for name, dtype in frame.dtypes.items():
        if pd.api.types.is_bool_dtype(dtype):
            role = "categorical"
        elif pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype):
            role = "numeric"
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            role = "categorical"
        else:
            raise ValueError(f"Column {name!r} has unsupported dtype {dtype}; encode dates/complex values explicitly")
        roles[role].append(name)
    if not any(roles.values()):
        raise ValueError("Select at least one input column")
    return roles


def _numeric_columns(frame):
    return infer_columns(frame)["numeric"]


def _categorical_columns(frame):
    return infer_columns(frame)["categorical"]


def auto(*, scale=True, sparse=True):
    """A native transformer whose column selectors resolve only at training fit."""
    transformer = require("sklearn.compose", "classical").ColumnTransformer
    return transformer([
        ("numeric", numeric(scale=scale), _numeric_columns),
        ("categorical", categorical(sparse=sparse), _categorical_columns),
    ], remainder="drop", sparse_threshold=1.0 if sparse else 0.0)


def ordinal():
    """Ordinal categorical pipeline; reserve -1 for unknown and -2 for missing."""
    Pipeline = require("sklearn.pipeline", "classical").Pipeline
    encoder = require("sklearn.preprocessing", "classical").OrdinalEncoder
    normalize = require("sklearn.preprocessing", "classical").FunctionTransformer
    return Pipeline([("types", normalize(_object_array, feature_names_out="one-to-one")),
                     ("encode", encoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                        encoded_missing_value=-2))])


def _object_array(values):
    return require("numpy", "classical").asarray(values, dtype=object)


def target_encoder(*, task="classification", cv=5, seed=0, **parameters):
    """Use in a pipeline: native fit_transform performs internal cross-fitting."""
    import re
    from numbers import Integral
    if task not in {"classification", "regression"}:
        raise ValueError("Choose classification or regression for target encoding")
    if task == "regression":
        parameters.setdefault("target_type", "continuous")
    encoder = require("sklearn.preprocessing", "classical").TargetEncoder
    version = tuple(map(int, re.findall(r"\d+", require("sklearn", "classical").__version__)[:2]))
    if version >= (1, 9):
        if isinstance(cv, Integral) and not isinstance(cv, bool):
            selection = require("sklearn.model_selection", "classical")
            cls = selection.KFold if parameters.get("target_type") == "continuous" else selection.StratifiedKFold
            cv = cls(n_splits=cv, shuffle=True, random_state=seed)
        return encoder(cv=cv, **parameters)
    return encoder(cv=cv, random_state=seed, **parameters)


def text(kind="tfidf", **parameters):
    vectorizers = require("sklearn.feature_extraction.text", "classical")
    if kind == "hashing":
        return vectorizers.HashingVectorizer(**parameters)
    if kind not in {"tfidf", "count"}:
        raise ValueError("Choose tfidf, count or hashing text features")
    parameters.setdefault("max_features", 20000)
    cls = vectorizers.TfidfVectorizer if kind == "tfidf" else vectorizers.CountVectorizer
    return cls(**parameters)


def polynomial(*, degree=2, interaction_only=False, include_bias=False):
    return require("sklearn.preprocessing", "classical").PolynomialFeatures(
        degree=degree, interaction_only=interaction_only, include_bias=include_bias)
