import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.utils.validation import check_is_fitted


def group_keys(groups,count):
    values=np.asarray(groups,dtype=object)
    if values.ndim!=1 or len(values)!=count: raise ValueError('Provide one group per row')
    result=[]
    for value in values.tolist():
        if value is None or not isinstance(value,(str,int,float)) or isinstance(value,float) and not np.isfinite(value): raise ValueError('Group labels must be finite scalar strings/numbers')
        result.append((type(value).__name__,value))
    return result


def positive_weights(weights,count):
    values=np.asarray(weights,dtype=float)
    if values.shape!=(count,) or not np.isfinite(values).all() or (values<=0).any(): raise ValueError('Likelihood ratios must be finite and strictly positive')
    return values


class _Intervals(RegressorMixin,BaseEstimator):
    def __init__(self,estimator,*,alpha=.1): self.estimator,self.alpha=estimator,alpha
    def fit(self,X,y): raise ValueError('Use nil.uncertainty calibration with an already fitted independent model')
    def predict(self,X):
        check_is_fitted(self,'estimator_')
        values=np.asarray(self.estimator_.predict(X),dtype=float)
        if not np.isfinite(values).all(): raise ValueError('Nonfinite regression predictions')
        return values


class MultioutputIntervals(_Intervals):
    def interval(self,X):
        predicted=self.predict(X)
        if predicted.ndim!=2 or predicted.shape[1]!=len(self.scales_): raise ValueError('Output schema differs')
        width=self.radius_*self.scales_
        return np.stack((predicted-width,predicted+width),axis=-1)


class GroupIntervals(_Intervals):
    def interval(self,X,groups):
        predicted=self.predict(X); keys=group_keys(groups,len(predicted))
        if predicted.ndim!=1: raise ValueError('Expected one regression output')
        if any(key not in self.radii_ for key in keys): raise ValueError('Unknown calibration group; no conditional guarantee is available')
        widths=np.asarray([self.radii_[key] for key in keys])
        return np.column_stack((predicted-widths,predicted+widths))


class WeightedIntervals(_Intervals):
    def interval(self,X,test_likelihood_ratios):
        predicted=self.predict(X); weights=positive_weights(test_likelihood_ratios,len(predicted))
        if predicted.ndim!=1: raise ValueError('Expected one regression output')
        # Include the test point's mass at +infinity, as required for finite-sample validity.
        cumulative=np.cumsum(self.weights_); threshold=(1-self.alpha)*(cumulative[-1]+weights)
        indices=np.searchsorted(cumulative,threshold,side='left')
        widths=np.asarray([self.scores_[index] if index<len(self.scores_) else np.inf for index in indices])
        return np.column_stack((predicted-widths,predicted+widths))


class BlockIntervals(_Intervals):
    def __init__(self,estimator,*,block_size,alpha=.1):
        super().__init__(estimator,alpha=alpha); self.block_size=block_size
    def interval(self,X):
        predicted=self.predict(X)
        if predicted.shape!=(self.block_size,): raise ValueError('Predict exactly one complete block at a time')
        return np.column_stack((predicted-self.radius_,predicted+self.radius_))
