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
        super().fit(np.concatenate(X), np.concatenate(y))

    def transform(self, X):
        transformed = X
        for sample_index in range(X.shape[0]):
            transformed[sample_index, :, :] = super().transform(X[sample_index, :, :])
        return transformed

    #pylint: disable=arguments-differ
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
