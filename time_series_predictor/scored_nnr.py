"""
scored_nnr
"""

from skorch import NeuralNetRegressor
import torch
import numpy as np

def sample_predict(model, inp):
    """Run predictions

    :param model: object to call predict method upon
    :param inp: input
    """
    return np.squeeze(model.predict(inp[np.newaxis, :, :]), axis=0)

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
        net_out = sample_predict(self, X)
        return self.criterion()(
            torch.Tensor(y).to(self.device),
            torch.Tensor(net_out).to(self.device))
