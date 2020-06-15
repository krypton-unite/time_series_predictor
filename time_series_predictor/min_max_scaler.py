"""
min_max_scaler
"""

import torch
import numpy as np
from .custom_sklearn.base import TransformerMixin
from .custom_sklearn import BaseEstimator

class MinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Min Max Scaler class
    """
    def __init__(self):
        self.max = None
        self.min = None

    # pylint: disable=invalid-name
    def fit(self, X: torch.Tensor, *_args, **_kwargs):
        """Compute the minimum and maximum to be used for later scaling.

        :param input_matrix: input matrix
        :param axis: None or int or tuple of ints, optional Axis or axes along which to operate.
        """
        self.max = X.max() # np.max(X, axis=(0, 1))
        self.min = X.min() # np.min(X, axis=(0, 1))
        return self

    # pylint: disable=invalid-name
    def fit_transform(self, X: torch.Tensor, y: torch.Tensor = None, **fit_params):
        """Fit to data, then transform it.

        :param input_matrix: input matrix
        :returns: transformed matrix
        """
        return self.fit(X, y, **fit_params).transform(X)

    # pylint: disable=invalid-name
    def transform(self, X: torch.Tensor):
        """Scale features of input_matrix according to feature_range.

        :param input_matrix: input matrix
        :returns: transformed matrix
        """
        return (X - self.min) / (self.max - self.min + np.finfo(float).eps)

    def inverse_transform(self, transformed: torch.Tensor):
        """
        inverse_transform

        :param transformed: transformed input
        :returns: inverse transformed
        """
        return self.min + transformed*(self.max - self.min)
