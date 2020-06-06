"""
time_series_predictor script
"""
import warnings

import numpy as np
import psutil
import torch
from skorch import NeuralNetRegressor
from .time_series_dataset import TimeSeriesDataset

# Show switch to cpu warning
warnings.filterwarnings("default", category=ResourceWarning)

# pylint: disable=too-many-instance-attributes
class TimeSeriesPredictor:
    """Network agnostic time series predictor class

    :param learning_rate:
    :param epochs:
    """
    def __init__(self, **neural_net_regressor_params):
        self.neural_net_regressor = None
        self.dataset = None
        self.cpu_count = psutil.cpu_count(logical=False)
        if 'device' in neural_net_regressor_params:
            device = neural_net_regressor_params.get('device')
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        neural_net_regressor_params.setdefault('device', device)
        self.neural_net_regressor_params = neural_net_regressor_params

    def make_future_dataframe(self, *args, **kwargs):
        # pylint: disable=anomalous-backslash-in-string
        """make_future_dataframe

        :param \*args: variable length unnamed args list
        :param \*\*kwargs: variable length named args list
        :returns: future dataframe
        """
        return self.dataset.make_future_dataframe(*args, **kwargs)

    def forecast(self, *args, **kwargs):
        # pylint: disable=anomalous-backslash-in-string
        """Future forecast

        :param \*args: variable length unnamed args list
        :param \*\*kwargs: variable length named args list
        """
        return self.predict(self.make_future_dataframe(*args, **kwargs))

    def predict(self, inp):
        """Run predictions

        :param inp: input
        """
        return np.squeeze(self.neural_net_regressor.predict(inp[np.newaxis, :, :]), axis=0)

    def fit(self, dataset: TimeSeriesDataset, net, **fit_params):
        """Fit selected network

        :param dataset: dataset to fit on
        :param net: network to use
        :param loss_function: optional loss function to use
        """
        print(f"Using device {self.neural_net_regressor_params.get('device')}")
        self.dataset = dataset
        self.neural_net_regressor = NeuralNetRegressor(
            net,
            **self.neural_net_regressor_params
        )
        try:
            self.neural_net_regressor.fit(dataset.x, dataset.y, **fit_params)
        except RuntimeError as err:
            if str(err).startswith('CUDA out of memory.'):
                warnings.warn(
                    '\nSwitching device to cpu to workaround CUDA out of memory problem.',
                    ResourceWarning)
                self.neural_net_regressor_params.setdefault('device', 'cpu')
                self.neural_net_regressor = NeuralNetRegressor(
                    net,
                    **self.neural_net_regressor_params
                )
                self.neural_net_regressor.fit(dataset.x, dataset.y, **fit_params)
            raise

    def compute_loss(self, dataloader):
        """Compute the loss of a network on a given dataset.

        Does not compute gradient.

        :param dataloader: iterator on the dataset.
        :returns: loss with no grad.
        """
        dataloader_length = len(dataloader)
        loss = np.empty(dataloader_length)
        device = self.neural_net_regressor_params.get('device')
        for idx_batch, (inp, out) in enumerate(dataloader):
            net_out = self.neural_net_regressor.predict(inp.to(device))
            loss[idx_batch] = self.neural_net_regressor.criterion()(out.to(device), torch.Tensor(net_out).to(device))

        return loss

    def compute_mean_loss(self, dataloader):
        """Compute the mean loss of a network on a given dataset.

        Does not compute gradient.

        :param dataloader: iterator on the dataset.
        :returns: mean loss with no grad.
        """
        return np.mean(self.compute_loss(dataloader))

    # def score(self):
    #     return self.compute_mean_loss(self.dataloader)

    # def get_training_dataframe(self):
    #     """get_training_dataframe

    #     :returns: training dataframe
    #     """
    #     return [inp for (inp, out) in self.dataloader]
