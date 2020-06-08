"""
min_max_scaler
"""

import numpy as np

class MinMaxScaler:
    """
    Min Max Scaler class
    """
    def __init__(self):
        self.max = None
        self.min = None

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling.

        :param input_matrix: input matrix
        :param axis: None or int or tuple of ints, optional Axis or axes along which to operate.
        """
        self.max = np.max(X, axis=(0, 1))
        self.min = np.min(X, axis=(0, 1))

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        :param input_matrix: input matrix
        :returns: transformed matrix
        """
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        """Scale features of input_matrix according to feature_range.

        :param input_matrix: input matrix
        :returns: transformed matrix
        """
        return (X - self.min) / (self.max - self.min + np.finfo(float).eps)
