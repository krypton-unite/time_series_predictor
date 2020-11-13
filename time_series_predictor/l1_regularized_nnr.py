"""
L1 regularized NNR

https://skorch.readthedocs.io/en/stable/user/neuralnet.html?highlight=NeuralNet%20get_loss#subclassing-neuralnet
"""
from skorch import NeuralNetRegressor


class L1RegularizedNNR(NeuralNetRegressor):
    r""" L1 regularization
    
    .. math:: L_{loss}=\left \| y-\hat{y} \right \|^2+\lambda | W |
    
    L1 regularization makes the weight vector sparse during the optimization process.

    The optimizer in PyTorch can only implement L2 regularization, and L1 regularization
    needs to be implemented manually, that is the purpose of this class. 

    .. note:: This example also regularizes the biases, which you typically
        don't need to do.
    """

    def __init__(self, *args, lambda1=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda1 = lambda1

    def get_loss(self, y_pred, y_true, X=None, training=False):
        loss = super().get_loss(y_pred, y_true, X=X, training=training)
        loss += self.lambda1 * sum([w.abs().sum()
                                    for w in self.module_.parameters()])
        return loss
