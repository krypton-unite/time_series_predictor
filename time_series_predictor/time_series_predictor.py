"""
time_series_predictor script
"""
import warnings

import psutil
import torch
from sklearn.pipeline import Pipeline
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback

from .min_max_scaler import MinMaxScaler as Scaler
from .sklearn import TransformedTargetRegressor, sample_predict
from .time_series_dataset import TimeSeriesDataset

# Show switch to cpu warning
warnings.filterwarnings("default", category=ResourceWarning)

class InputShapeSetter(Callback):
    """InputShapeSetter
    dynamically set the input size of the PyTorch module based on the data

    Typically, it’s up to the user to determine the shape of the input data when defining
    the PyTorch module. This can sometimes be inconvenient, e.g. when the shape is only
    known at runtime. E.g., when using :class:`sklearn.feature_selection.VarianceThreshold`,
    you cannot know the number of features in advance. The best solution would be to set
    the input size dynamically.
    """
    def on_train_begin(self, net, X=None, y=None, **kwargs):
        net.set_params(module__input_dim=X.shape[-1],
                       module__output_dim=y.shape[-1])

class TimeSeriesPredictor:
    """Network agnostic time series predictor class

    Parameters
    ----------
    **neural_net_regressor_params: skorch NeuralNetRegressor parameters.
    """

    def __init__(self, net, **neural_net_regressor_params):
        self.dataset = None
        self.cpu_count = psutil.cpu_count(logical=False)
        if 'device' in neural_net_regressor_params:
            device = neural_net_regressor_params.get('device')
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        neural_net_regressor_params.setdefault('device', device)

        self.ttr = TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ('input scaler', Scaler()),
                    (
                        'regressor', NeuralNetRegressor(
                            net,
                            callbacks=[InputShapeSetter()],
                            **neural_net_regressor_params
                        )
                    )
                ]
            ),
            transformer=Scaler()
        )

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

    def sample_forecast(self, *args, **kwargs):
        """Future forecast

        Parameters
        ----------
        *args: variable length unnamed args list
        **kwargs: variable length named args list

        :returns: future forecast
        """
        return self.sample_predict(self.make_future_dataframe(*args, **kwargs))

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
        return self.ttr.predict(inp)

    def sample_predict(self, inp):
        """Run predictions

        :param inp: input
        """
        return sample_predict(self.ttr, inp)

    def fit(self, dataset: TimeSeriesDataset, **fit_params):
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
        try:
            self.ttr.fit(dataset.x, dataset.y, **fit_params)
        except RuntimeError as err:
            if str(err).startswith('CUDA out of memory.'):
                warnings.warn(
                    '\nSwitching device to cpu to workaround CUDA out of memory problem.',
                    ResourceWarning)
                self.neural_net_regressor_params.setdefault('device', 'cpu')
                self.ttr.regressor['regressor'] = NeuralNetRegressor(
                    self.ttr.regressor['regressor'].module,
                    callbacks=[InputShapeSetter()],
                    **self.neural_net_regressor_params
                )
                self.ttr.fit(dataset.x, dataset.y, **fit_params)
            raise

    def score(self, dataset):
        """Compute the mean r2_score of a network on a given dataset.

        :param dataset: dataset to evaluate.
        :returns: mean r2_score.
        """
        losses = [
            self.ttr.score(
                inp,
                out
            ) for (inp, out) in dataset
        ]
        assert all(loss <= 1 for loss in losses)
        device = self.neural_net_regressor_params.get('device')
        return torch.mean(torch.Tensor(losses).to(device)).cpu().numpy().take(0)
