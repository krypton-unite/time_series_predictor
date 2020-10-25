"""
time_series_predictor script
"""
import queue
import threading
import warnings

import psutil
import torch
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback, EarlyStopping
from time_series_dataset import TimeSeriesDataset

from sklearn.pipeline import Pipeline

from .min_max_scaler import MinMaxScaler as Scaler
from .sklearn import TransformedTargetRegressor, sample_predict

# Show switch to cpu warning
warnings.filterwarnings("default", category=ResourceWarning)

def _get_nnr(net, early_stopping: EarlyStopping = None, **neural_net_regressor_params):
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
    callbacks = [InputShapeSetter()]
    if early_stopping is not None:
        callbacks.append(early_stopping)
    return NeuralNetRegressor(
        net,
        callbacks=callbacks,
        **neural_net_regressor_params
    )

class TimeSeriesPredictor:
    """Network agnostic time series predictor class

    Parameters
    ----------
    **neural_net_regressor_params: skorch NeuralNetRegressor parameters.
    early_stopping: torch.callbacks.EarlyStopping object

    """

    def __init__(self, net, early_stopping: EarlyStopping = None, **neural_net_regressor_params):
        self.dataset = None
        self.early_stopping = early_stopping
        self.cpu_count = psutil.cpu_count(logical=True)
        if 'train_split' in neural_net_regressor_params:
            train_split = neural_net_regressor_params.get('train_split')
            if train_split is None and early_stopping is not None:
                if early_stopping.monitor is 'valid_loss':
                    raise ValueError('Select a valid train_split or disable early_stopping! A valid train_split needs to be selected when valid_loss monitor is selected as early stopping criteria.')
        if 'device' in neural_net_regressor_params:
            device = neural_net_regressor_params.get('device')
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        neural_net_regressor_params.setdefault('device', device)

        self.ttr = TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ('input scaler', Scaler()),
                    ('regressor', _get_nnr(net, early_stopping, **neural_net_regressor_params))
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
                self.ttr.regressor['regressor'] = _get_nnr(
                    self.ttr.regressor['regressor'].module,
                    self.early_stopping,
                    **self.neural_net_regressor_params
                )
                self.ttr.fit(dataset.x, dataset.y, **fit_params)
            raise

    def score(self, dataset):
        """Compute the mean r2_score of a network on a given dataset.

        :param dataset: dataset to evaluate.
        :returns: mean r2_score.

        """
        dataset_length = len(dataset)
        if dataset_length == 1:
            # losses = [
            #     self.ttr.score(
            #         inp,
            #         out
            #     ) for (inp, out) in dataset
            # ]
            inp, out = dataset[0]
            return self.ttr.score(inp, out)
        scores_to_calculate = queue.Queue()
        locks = [threading.Lock() for _ in range(2)]
        for idx, (inp, out) in enumerate(dataset):
            scores_to_calculate.put([idx, (inp, out)])
        output_list = []
        score_calculators = []
        for _ in range(dataset_length if dataset_length < self.cpu_count else self.cpu_count):
            score_calculator = ScoreCalculator(
                scores_to_calculate,
                locks,
                self.ttr.score,
                output_list
            )
            score_calculators.append(score_calculator)
        for score_calculator in score_calculators:
            score_calculator.start()
        for score_calculator in score_calculators:
            score_calculator.join()
        losses = [loss[1] for loss in output_list]
        # assert all(loss <= 1 for loss in losses)

        device = self.neural_net_regressor_params.get('device')
        return torch.mean(torch.Tensor(losses).to(device)).cpu().numpy().take(0)

class ScoreCalculator(threading.Thread):
    """ScoreCalculator
    Spread computational effort across cpus
    """
    def __init__(self, scores_to_calculate, locks, score_method, output_list):
        super().__init__()
        # print("Web Crawler Worker Started: {}".format(threading.current_thread()))
        self.scores_to_calculate = scores_to_calculate
        self.locks = locks
        self.score_method = score_method
        self.output_list = output_list

    def run(self):
        while True:
            if self.scores_to_calculate.empty():
                return
            self.locks[0].acquire()
            score_to_calculate = self.scores_to_calculate.get()
            self.locks[0].release()

            r2_score = self.score_method(*score_to_calculate[1])
            record = [score_to_calculate[0], r2_score]

            self.locks[1].acquire()
            self.output_list.append(record)
            self.locks[1].release()

            self.locks[0].acquire()
            self.scores_to_calculate.task_done()
            self.locks[0].release()
