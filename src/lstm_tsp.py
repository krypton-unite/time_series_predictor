"""
lstm_tsp
"""
import torch

from time_series_predictor import TimeSeriesPredictor

from .model import BenchmarkLSTM


class LSTMTimeSeriesPredictor(TimeSeriesPredictor):
    """
    TransformerTimeSeriesPredictor

    :param learning_rate: learning rate of the optimizer
    :param epochs: number of epochs to spend fitting
    :param hidden_dim: latent dimension
    :param num_layers: number of LSTM layers
    """
    def __init__(self,
                 hidden_dim=100,
                 num_layers=3,
                 **neural_net_regressor_params):
        super().__init__(**neural_net_regressor_params)
        self.hidden_dim = hidden_dim    # Number of neurons in hidden layers
        self.num_layers = num_layers    # Number of layers

    # pylint: disable=arguments-differ
    def fit(self, dataset, **fit_params):
        """fit

        :param dataset: dataset to fit LSTM on
        :param \*\*fit_params: Additional parameters passed to the forward method of the module and to the self.train_split call.
        :returns: loss history during fitting
        """
        """Initialize and fit the module.
        If the module was already initialized, by calling fit, the
        module will be re-initialized (unless ``warm_start`` is True).
        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset
          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.
        y : target data, compatible with skorch.dataset.Dataset
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.
        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.
        """
        d_input = dataset.get_x_shape()[2]     # From dataset
        d_output = dataset.get_y_shape()[2]    # From dataset
        net = BenchmarkLSTM(input_dim=d_input, hidden_dim=self.hidden_dim,
                            output_dim=d_output, num_layers=self.num_layers)
        return super().fit(dataset, net, **fit_params)
