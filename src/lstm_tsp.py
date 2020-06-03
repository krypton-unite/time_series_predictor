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
                 learning_rate=1e-2,
                 epochs=5,
                 hidden_dim=100,
                 num_layers=3):
        super().__init__(learning_rate, epochs)
        self.hidden_dim = hidden_dim    # Number of neurons in hidden layers
        self.num_layers = num_layers    # Number of layers

    # pylint: disable=arguments-differ
    def fit(self, dataset, loss_function=torch.nn.MSELoss()):
        """fit

        :param dataset: dataset to fit LSTM on
        :param loss_function: optional loss function to use
        :returns: loss history during fitting
        """
        d_input = dataset.get_x_shape()[2]     # From dataset
        d_output = dataset.get_y_shape()[2]    # From dataset
        net = BenchmarkLSTM(input_dim=d_input, hidden_dim=self.hidden_dim,
                            output_dim=d_output, num_layers=self.num_layers)
        return super().fit(dataset, net, loss_function=loss_function)
