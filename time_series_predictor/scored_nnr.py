"""
scored_nnr
"""

from skorch import NeuralNetRegressor
import torch
import numpy as np

class ScoredNnr(NeuralNetRegressor):
    """
    class ScoredNnr
    """

    def __init(
            self,
            module,
            *args,
            criterion=torch.nn.MSELoss,
            **kwargs):
        self.device = kwargs.get('device')
        super().__init__(module, *args, criterion=criterion, **kwargs)

    def score(self, X, y, sample_weight=None):
        net_out = self.predict(X)
        return self.criterion()(
            torch.Tensor(y).to(self.device),
            torch.Tensor(net_out).to(self.device))

    def predict(self, X):
        """Run predictions

        :param X: input
        """
        return np.squeeze(super().predict(X[np.newaxis, :, :]), axis=0)
