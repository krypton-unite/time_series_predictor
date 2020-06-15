"""
tensor_nnr
"""

import torch
from skorch import NeuralNetRegressor

class TensorNeuralNetRegressor(NeuralNetRegressor):
    """specialized NeuralNetRegressor class that returns a Torch tensor as prediction output

    Args:
        NeuralNetRegressor ([type]): [description]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = kwargs.get('device')
    def predict(self, X):
        return torch.tensor(super().predict(X), dtype=torch.float64).to(self.device)
