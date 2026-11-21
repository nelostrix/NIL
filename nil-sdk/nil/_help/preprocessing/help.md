# preprocessing

Build leakage-resistant native preprocessing and feature-selection pipelines.

## Usage

```python
import nil

preprocess = nil.preprocessing.tabular(numeric_columns=[0], categorical_columns=[1])
model = nil.pipeline(nil.model("logistic_regression"), preprocess=preprocess)
X = [[1., "a"], [2., "b"], [3., "a"], [4., "b"]]
result = nil.fit(model, X, [0, 1, 0, 1])
print(result.predict([[5., "unseen"]]))
```

## Parameters

numeric/categorical column names or positions; scale=False or a scaler name; sparse=True; pipeline(model, preprocess=..., select=...). select_features(k=..., task=...) supplies task-appropriate scoring.
`auto()` chooses numeric/categorical DataFrame columns from training dtypes at fit.
`infer_columns(frame)` previews those roles. `ordinal()` reserves -1 for unknown
and -2 for missing categories. `text('tfidf'/'count'/'hashing', **parameters)`
creates a native vectorizer; tfidf/count default max_features to 20000.
`polynomial(degree=2)` supplies a native feature expansion.
`target_encoder(task='classification'/'regression', cv=5, seed=0, **parameters)`
creates a cross-fitted native encoder, with version-appropriate CV configuration.

## Output

Native sklearn pipelines/ColumnTransformers; only training fits preprocessing.

## Requirements

Install `nil-sdk[classical]` for sklearn operations. Named DataFrame inputs additionally require pandas. Backends load only when called.
Automatic columns use `nil-sdk[tabular]` for pandas and sklearn.

## Limitations

Use numpy.nan for missing values. Standard/robust sparse scaling does not center. MinMaxScaler requires dense input. Dropped columns are intentional; class-specific valid values still matter. Never fit transforms/selectors on held-out data.
Automatic inference supports numeric and categorical/string/bool dtypes; dates
and complex values fail with an explicit request for feature engineering. Pass
feature columns only, with targets kept separate. Text columns are categorized
by auto; use an explicit text vectorizer for free text. Normalize pd.NA/None
sentinels when necessary for native estimators. Polynomial expansions can grow
rapidly; reduce columns/degree before fitting. Put target encoders inside the
pipeline so fit_transform cross-fits training encodings; fit().transform() is
different. The helper's default inner folds are not group/time aware. For those
problems use a supported native configured splitter/version or avoid target
encoding; outer folds alone do not make its inner encodings safe.
