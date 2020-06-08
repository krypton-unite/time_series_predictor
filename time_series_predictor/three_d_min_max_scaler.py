"""
three_d_min_max_scaler
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ThreeDMinMaxScaler(MinMaxScaler):
    """ThreeDMinMaxScaler

    Extend MinMaxScaler to allow 3 dimensional input
    """
    def __init__(self, *args, feature_range=(0, 1), copy=True):
        super().__init__(*args, feature_range=feature_range, copy=copy)

    def fit(self, X, y=None):
        super().fit(np.concatenate([X[sample_index, :, :] for sample_index in range(X.shape[0])]))

    def transform(self, X):
        output = X
        for sample_index in range(X.shape[0]):
            output[sample_index, :, :] = super().transform(X[sample_index, :, :])
        return output

    #pylint: disable=arguments-differ
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
