# sklearn

Access scikit-learn's public API through NIL. This namespace mirrors the installed
backend instead of copying its algorithms or wrapping each prediction.

## Usage

```python
import nil
from nil.sklearn.linear_model import LogisticRegression
from nil.sklearn.feature_extraction.text import TfidfVectorizer
from nil.sklearn.pipeline import make_pipeline
from nil.sklearn.model_selection import GridSearchCV

model = make_pipeline(TfidfVectorizer(), LogisticRegression(random_state=7))
search = GridSearchCV(model, {"logisticregression__C": [0.5, 1]}, cv=2, n_jobs=1)
texts = ["good happy", "great happy", "good bright", "great bright",
         "bad sad", "awful sad", "bad dark", "awful dark"]
labels = [1, 1, 1, 1, 0, 0, 0, 0]
search.fit(texts, labels)
print(search.predict(["great bright", "bad dark"]))
print(nil.sklearn.__version__)
```

## Parameters

Classes/functions preserve the installed scikit-learn signatures, parameter
names, defaults, validation and errors. Imports follow the same layout with
`nil.` before `sklearn`, including nested public modules. Root helpers such as
`nil.sklearn.clone` and `nil.sklearn.config_context` are also available.

## Output

Native scikit-learn objects. Estimator classes retain their original identity,
cloning, fitted attributes and serialization paths. They work with NIL's
classical helpers where that helper supports the estimator's task. Public
modules are namespace proxies; their classes and functions are unchanged.

## Requirements

Explicitly install `nil-sdk[classical]` (or `.[classical]` from source) in the
environment that will run the models. The base NIL wheel still has zero runtime
dependencies. Importing `nil.sklearn` alone loads no compute engine; requesting a
backend API/submodule loads it. Nothing is automatically installed or downloaded
by the mirror. Dataset fetch functions retain their native behavior when called.

## Limitations

This is API access, not an independent NIL numerical implementation. Backend
versions determine which APIs exist. Private underscore modules are excluded.
The mirror does not change native threading, memory use, optional plotting/data
dependencies or dataset download behavior. Native CV/search budgets/defaults
remain native; use NIL's bounded validation helpers when those limits matter.
NIL's evaluator supports multilabel/multioutput classification with aggregate
and per-output metrics; probability-specific metrics depend on native output
shape. Scikit-learn does not provide
every specialized forecasting, survival, ranking or deep learning workflow.

See [the upstream API reference](https://scikit-learn.org/stable/api/index.html).
