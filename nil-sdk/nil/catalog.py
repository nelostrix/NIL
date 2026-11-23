"""A shallow model catalog. Listing models never imports a compute backend."""

from nil.runtime import require

# name: (group, module, constructor, optional dependency extra)
_MODELS = {
    "lag_forecaster": ("forecasting", "nil._forecast", "LagForecaster", "classical"),
    "query_ranker": ("ranking", "nil._ranking", "QueryRanker", "lightgbm"),
    "item_cosine": ("recommendation", "nil._recommendation", "ItemRecommender", "classical"),
    "cox_ph": ("survival", "nil._survival", "CoxModel", "survival"),
    "kaplan_meier": ("survival", "nil._survival", "KaplanMeier", "survival"),
    "lightgbm_classifier": ("classification", "lightgbm", "LGBMClassifier", "lightgbm"),
    "lightgbm_regressor": ("regression", "lightgbm", "LGBMRegressor", "lightgbm"),
    "catboost_classifier": ("classification", "catboost", "CatBoostClassifier", "catboost"),
    "catboost_regressor": ("regression", "catboost", "CatBoostRegressor", "catboost"),
    "linear_regression": ("regression", "sklearn.linear_model", "LinearRegression", "classical"),
    "ridge_regression": ("regression", "sklearn.linear_model", "Ridge", "classical"),
    "lasso_regression": ("regression", "sklearn.linear_model", "Lasso", "classical"),
    "decision_tree_regressor": ("regression", "sklearn.tree", "DecisionTreeRegressor", "classical"),
    "random_forest_regressor": ("regression", "sklearn.ensemble", "RandomForestRegressor", "classical"),
    "bagging_regressor": ("regression", "sklearn.ensemble", "BaggingRegressor", "classical"),
    "gradient_boosting_regressor": ("regression", "sklearn.ensemble", "HistGradientBoostingRegressor", "classical"),
    "xgboost_regressor": ("regression", "xgboost", "XGBRegressor", "boosting"),
    "logistic_regression": ("classification", "sklearn.linear_model", "LogisticRegression", "classical"),
    "decision_tree_classifier": ("classification", "sklearn.tree", "DecisionTreeClassifier", "classical"),
    "random_forest_classifier": ("classification", "sklearn.ensemble", "RandomForestClassifier", "classical"),
    "bagging_classifier": ("classification", "sklearn.ensemble", "BaggingClassifier", "classical"),
    "gradient_boosting_classifier": ("classification", "sklearn.ensemble", "HistGradientBoostingClassifier", "classical"),
    "xgboost_classifier": ("classification", "xgboost", "XGBClassifier", "boosting"),
    "kmeans": ("unsupervised", "sklearn.cluster", "KMeans", "classical"),
    "pca": ("unsupervised", "sklearn.decomposition", "PCA", "classical"),
    "mlp": ("neural_networks", "nil.nn", "mlp", "torch"),
    "elastic_net": ("regression", "sklearn.linear_model", "ElasticNet", "classical"),
    "sgd_regressor": ("regression", "sklearn.linear_model", "SGDRegressor", "classical"),
    "huber_regressor": ("regression", "sklearn.linear_model", "HuberRegressor", "classical"),
    "ransac_regressor": ("regression", "sklearn.linear_model", "RANSACRegressor", "classical"),
    "knn_regressor": ("regression", "sklearn.neighbors", "KNeighborsRegressor", "classical"),
    "svr": ("regression", "sklearn.svm", "SVR", "classical"),
    "linear_svr": ("regression", "sklearn.svm", "LinearSVR", "classical"),
    "voting_regressor": ("regression", "sklearn.ensemble", "VotingRegressor", "classical"),
    "stacking_regressor": ("regression", "sklearn.ensemble", "StackingRegressor", "classical"),
    "sgd_classifier": ("classification", "sklearn.linear_model", "SGDClassifier", "classical"),
    "knn_classifier": ("classification", "sklearn.neighbors", "KNeighborsClassifier", "classical"),
    "svc": ("classification", "sklearn.svm", "SVC", "classical"),
    "linear_svc": ("classification", "sklearn.svm", "LinearSVC", "classical"),
    "gaussian_nb": ("classification", "sklearn.naive_bayes", "GaussianNB", "classical"),
    "multinomial_nb": ("classification", "sklearn.naive_bayes", "MultinomialNB", "classical"),
    "bernoulli_nb": ("classification", "sklearn.naive_bayes", "BernoulliNB", "classical"),
    "voting_classifier": ("classification", "sklearn.ensemble", "VotingClassifier", "classical"),
    "stacking_classifier": ("classification", "sklearn.ensemble", "StackingClassifier", "classical"),
    "calibrated_classifier": ("classification", "sklearn.calibration", "CalibratedClassifierCV", "classical"),
    "dbscan": ("unsupervised", "sklearn.cluster", "DBSCAN", "classical"),
    "agglomerative_clustering": ("unsupervised", "sklearn.cluster", "AgglomerativeClustering", "classical"),
    "gaussian_mixture": ("unsupervised", "sklearn.mixture", "GaussianMixture", "classical"),
    "truncated_svd": ("unsupervised", "sklearn.decomposition", "TruncatedSVD", "classical"),
    "isolation_forest": ("anomaly_detection", "sklearn.ensemble", "IsolationForest", "classical"),
    "one_class_svm": ("anomaly_detection", "sklearn.svm", "OneClassSVM", "classical"),
    "variance_threshold": ("feature_selection", "sklearn.feature_selection", "VarianceThreshold", "classical"),
    "select_k_best": ("feature_selection", "sklearn.feature_selection", "SelectKBest", "classical"),
    "select_from_model": ("feature_selection", "sklearn.feature_selection", "SelectFromModel", "classical"),
}
_EXTENSIONS = {}


def register(name, factory, *, group, help_path=None):
    """Register a custom model constructor without calling it or loading its backend.

    Registration is process-local and explicit. Existing model names cannot be
    replaced. Put optional dependency imports inside the factory to keep loading lazy.
    """
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError("Model name must be a Python identifier, for example my_regressor")
    if not isinstance(group, str) or not group.isidentifier():
        raise ValueError("Model group must be a Python identifier, for example regression")
    if not callable(factory):
        raise TypeError("Model factory must be callable")
    if name in _MODELS or name in _EXTENSIONS:
        raise ValueError(f"Model {name!r} is already registered")
    _EXTENSIONS[name] = (group, factory, help_path)


def list_models(group=None):
    """Return supported constructor names, optionally filtered by their shallow group."""
    groups = {entry[0] for entry in _MODELS.values()} | {entry[0] for entry in _EXTENSIONS.values()}
    if group is not None and group not in groups:
        raise ValueError(f"Unknown model group {group!r}. Choose: {', '.join(sorted(groups))}")
    return [name for registry in (_MODELS, _EXTENSIONS)
            for name, entry in registry.items() if group is None or entry[0] == group]


def tree():
    """Return group -> model names. This is metadata, not a backend readiness check."""
    result = {}
    for name, (group, *_rest) in _MODELS.items():
        result.setdefault(group, []).append(name)
    for name, (group, _factory, _help_path) in _EXTENSIONS.items():
        result.setdefault(group, []).append(name)
    return result


def create(name, **parameters):
    """Construct the native model directly; backend parameters and errors remain explicit.

    Classical/boosting models expose their native fit/predict methods. Neural
    modules work with nil.train. This catalog does not add a universal trainer.
    """
    if not isinstance(name, str):
        raise ValueError("Model name must be a string; use nil.list_models() for supported names")
    if name in _EXTENSIONS:
        return _EXTENSIONS[name][1](**parameters)
    if name not in _MODELS:
        raise ValueError(f"Unknown model {name!r}. Use nil.list_models() to see supported names.")
    if name in {'lightgbm_classifier', 'lightgbm_regressor', 'catboost_classifier', 'catboost_regressor'}:
        from nil import boosting
        return getattr(boosting, name)(**parameters)
    _group, module, constructor, extra = _MODELS[name]
    if module.startswith("nil._"):
        require("sklearn",extra)
    return getattr(require(module, extra), constructor)(**parameters)
