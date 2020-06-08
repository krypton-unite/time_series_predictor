"""
time_series_predictor script
"""
import warnings

import numpy as np
import psutil
import torch
from sklearn.pipeline import Pipeline
from .scored_nnr import ScoredNnr
from .time_series_dataset import TimeSeriesDataset
from .min_max_scaler import MinMaxScaler as Scaler

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
        """make_future_dataframe

        Parameters
        ----------
        *args: variable length unnamed args list
        **kwargs: variable length named args list

        :returns: future dataframe
        """
        return self.dataset.make_future_dataframe(*args, **kwargs)

    def forecast(self, *args, **kwargs):
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
        return self.pipe.predict(inp)

    def _config_fit(self, net):
        self.pipe = Pipeline([
            ('input scaler', Scaler()),
            ('regressor', ScoredNnr(net, **self.neural_net_regressor_params))
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

    def score(self, dataset):
        """Compute the loss of a network on a given dataset.

        :param dataloader: iterator on the dataset.
        :returns: loss with no grad.
        """
        dataloader = self.pipe['regressor'].get_iterator(dataset)
        dataloader_length = len(dataloader)
        loss = np.empty(dataloader_length)
        for idx_batch, (inp, out) in enumerate(dataloader):
            loss[idx_batch] = self.pipe.score(np.squeeze(inp, axis=0), np.squeeze(out, axis=0))
        return np.mean(loss)
