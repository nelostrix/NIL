from dataclasses import dataclass
from nil.runtime import require
from nil._classical_common import unwrap, positive_int


@dataclass
class ShapExplanation:
    values: object
    base_values: object
    predictions: object
    feature_names: list
    standard_errors: object
    evaluations: int

    def to_dict(self):
        return {name: value.tolist() if hasattr(value, 'tolist') else value for name, value in vars(self).items()}


def permutation_shap(model, features, background, *, permutations, max_rows, max_background,
                     max_features, max_evaluations, max_dense_bytes, class_index, seed):
    np = require('numpy', 'classical'); sparse = require('scipy.sparse', 'classical')
    for name,value in [('permutations',permutations),('max_rows',max_rows),('max_background',max_background),('max_features',max_features),('max_evaluations',max_evaluations),('max_dense_bytes',max_dense_bytes)]: positive_int(value,name)
    model = unwrap(model)
    require('sklearn.utils.validation', 'classical').check_is_fitted(model)
    count = features.shape[0] if hasattr(features, 'shape') else len(features)
    backgrounds = background.shape[0] if hasattr(background, 'shape') else len(background)
    if not 0 < count <= max_rows or not 0 < backgrounds <= max_background:
        raise ValueError('Explanation rows/background exceed limits; explicitly select a representative subset')
    names = list(getattr(features, 'columns', getattr(model,'feature_names_in_',[])))
    if hasattr(features,'columns') and hasattr(background,'columns') and list(features.columns)!=list(background.columns):
        raise ValueError('Explanation/background feature columns must match in order')
    if hasattr(features,'columns') and hasattr(model,'feature_names_in_') and list(features.columns)!=list(model.feature_names_in_):
        raise ValueError('Explanation columns differ from the fitted model schema')
    if hasattr(model, 'steps') and len(model.steps)>1:
        transform, model = model[:-1], model.steps[-1][1]
        features, background = transform.transform(features), transform.transform(background)
        if features.shape[1]>max_features or background.shape[1]>max_features:
            raise ValueError('Transformed explanation feature/dense budget exceeded')
        try: names = list(transform.get_feature_names_out())
        except AttributeError: names = []
    def dense(values):
        shape = getattr(values,'shape',None)
        if shape is None: values=np.asarray(values); shape=values.shape
        if len(shape)!=2 or shape[1]>max_features or shape[0]*shape[1]*8>max_dense_bytes:
            raise ValueError('Transformed explanation feature/dense budget exceeded')
        array=np.asarray(values.toarray() if sparse.issparse(values) else values,dtype=float)
        if not np.isfinite(array).all(): raise ValueError('Explanations require finite numeric transformed inputs')
        return array
    x,bg=dense(features),dense(background)
    if len(x)!=count or len(bg)!=backgrounds: raise ValueError('Preprocessing must preserve explanation rows')
    if x.shape[1]!=bg.shape[1] or x.shape[1]==0: raise ValueError('Background feature schema differs')
    budget=len(bg)*(1+len(x)*permutations*x.shape[1])+len(x)
    if budget>max_evaluations: raise ValueError('Explanation exceeds max_evaluations before prediction')
    working_bytes=8*(x.size+2*bg.size+2*len(x)*permutations*x.shape[1])
    if working_bytes>max_dense_bytes: raise ValueError('Explanation working-array dense budget exceeded')
    def predict(rows):
        if class_index is None:
            predicted=np.asarray(model.predict(rows),dtype=float)
        else:
            probabilities=np.asarray(model.predict_proba(rows),dtype=float)
            if type(class_index) is not int or not 0<=class_index<probabilities.shape[1]: raise ValueError('Invalid class_index')
            predicted=probabilities[:,class_index]
        if predicted.shape!=(len(rows),) or not np.isfinite(predicted).all(): raise ValueError('Select one finite scalar model output for explanation')
        return predicted
    baseline=float(predict(bg).mean()); expected=predict(x)
    rng=np.random.default_rng(seed); samples=np.zeros((len(x),permutations,x.shape[1]))
    for row, values in enumerate(x):
        for repetition in range(permutations):
            hybrid=bg.copy(); previous=baseline
            for feature in rng.permutation(x.shape[1]):
                hybrid[:,feature]=values[feature]
                current=float(predict(hybrid).mean())
                samples[row,repetition,feature]=current-previous; previous=current
    attribution=samples.mean(axis=1)
    errors=samples.std(axis=1,ddof=1)/np.sqrt(permutations) if permutations>1 else None
    if not np.allclose(baseline+attribution.sum(axis=1),expected,rtol=1e-6,atol=1e-8): raise ValueError('Model predictions are not deterministic/additive across explanation calls')
    return ShapExplanation(attribution,np.full(len(x),baseline),expected,
        [str(name) for name in names] if len(names)==x.shape[1] else [f'x{i}' for i in range(x.shape[1])],errors,budget)
