"""
lstm_tsp
"""
from time_series_predictor import TimeSeriesPredictor

from .model import BenchmarkLSTM


class LSTMTimeSeriesPredictor(TimeSeriesPredictor):
    """
    TransformerTimeSeriesPredictor

    Parameters
    ----------
    hidden_dim: latent dimension
    num_layers: number of LSTM layers
    **neural_net_regressor_params: skorch NeuralNetRegressor parameters.
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
        """Initialize and fit the module.

        Parameters
        ----------
        dataset : dataset, compatible with skorch.dataset.Dataset
          Dataset to fit LSTM on
        **fit_params: dict
          Additional parameters passed to the forward method of the module and to the
          self.train_split call.
        """
        d_input = dataset.get_x_shape()[2]     # From dataset
        d_output = dataset.get_y_shape()[2]    # From dataset
        net = BenchmarkLSTM(input_dim=d_input, hidden_dim=self.hidden_dim,
                            output_dim=d_output, num_layers=self.num_layers)
        return super().fit(dataset, net, **fit_params)
