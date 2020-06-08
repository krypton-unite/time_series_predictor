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
        new_x = np.concatenate(X)
        if not y is None:
            super().fit(new_x, y=np.concatenate(y))
        super().fit(new_x)

    def transform(self, X):
        transformed = X
        for sample_index in range(X.shape[0]):
            transformed[sample_index, :, :] = self.two_d_transform(X[sample_index, :, :])
        return transformed

    def two_d_transform(self, inp):
        """two_d_transform

        Parameters
        ----------
        inp : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
        """
        return super().transform(inp)

    #pylint: disable=arguments-differ
    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)