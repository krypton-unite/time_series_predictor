"""
time_series_predictor script
"""
import warnings

import numpy as np
import psutil
import torch
from sklearn.pipeline import Pipeline
from skorch import NeuralNetRegressor
from .time_series_dataset import TimeSeriesDataset
from .three_d_min_max_scaler import ThreeDMinMaxScaler as Scaler
# from .min_max_scaler import MinMaxScaler as Scaler

# Show switch to cpu warning
warnings.filterwarnings("default", category=ResourceWarning)

class TimeSeriesPredictor:
    """Network agnostic time series predictor class

    Parameters
    ----------
    **neural_net_regressor_params: skorch NeuralNetRegressor parameters.
    """
    def __init__(self, **neural_net_regressor_params):
        self.pipe = None
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

        Parameters
        ----------
        *args: variable length unnamed args list
        **kwargs: variable length named args list

        :returns: future dataframe
        """
        return self.dataset.make_future_dataframe(self.pipe['input scaler'], *args, **kwargs)

    def forecast(self, *args, **kwargs):
        # pylint: disable=anomalous-backslash-in-string
        """Future forecast

        Parameters
        ----------
        *args: variable length unnamed args list
        **kwargs: variable length named args list

        :returns: future forecast
        """
        return self.predict(self.make_future_dataframe(*args, **kwargs))

    def predict(self, inp):
        """Run predictions

        :param inp: input
        """
        return np.squeeze(self.pipe['regressor'].predict(inp[np.newaxis, :, :]), axis=0)

    def _config_fit(self, net):
        self.pipe = Pipeline([
            ('input scaler', Scaler()),
            ('regressor', NeuralNetRegressor(net, **self.neural_net_regressor_params)),
            ('output scaler', Scaler()),
        ])

    def fit(self, dataset: TimeSeriesDataset, net, **fit_params):
        """Fit selected network

        Parameters
        ----------
        dataset: dataset to fit on
        net: network to use
        **fit_params: dict
          Additional parameters passed to the forward method of the module and to the
          self.train_split call.
        """
        print(f"Using device {self.neural_net_regressor_params.get('device')}")
        self.dataset = dataset
        self._config_fit(net)
        try:
            self.pipe.fit(dataset.x, dataset.y, **fit_params)
        except RuntimeError as err:
            if str(err).startswith('CUDA out of memory.'):
                warnings.warn(
                    '\nSwitching device to cpu to workaround CUDA out of memory problem.',
                    ResourceWarning)
                self.neural_net_regressor_params.setdefault('device', 'cpu')
                self._config_fit(net)
                self.pipe.fit(dataset.x, dataset.y, **fit_params)
            raise

    def compute_loss(self, dataloader):
        """Compute the loss of a network on a given dataset.

        :param dataloader: iterator on the dataset.
        :returns: loss with no grad.
        """
        dataloader_length = len(dataloader)
        loss = np.empty(dataloader_length)
        device = self.neural_net_regressor_params.get('device')
        for idx_batch, (inp, out) in enumerate(dataloader):
            net_out = self.pipe.predict(inp.to(device))
            loss[idx_batch] = self.pipe['regressor'].criterion()(
                out.to(device), torch.Tensor(net_out).to(device))

        return loss

    def compute_mean_loss(self, dataloader):
        """Compute the mean loss of a network on a given dataset.

        :param dataloader: iterator on the dataset.
        :returns: mean loss with no grad.
        """
        return np.mean(self.compute_loss(dataloader))
