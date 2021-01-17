"""
time_series_predictor script
"""
import queue
import tempfile
import threading
import warnings

import psutil
import torch
from IPython.utils import io
from skorch.callbacks import Callback, Checkpoint, EarlyStopping
from time_series_dataset import TimeSeriesDataset
# from tune_sklearn import TuneGridSearchCV

from sklearn.pipeline import Pipeline

from .l1_regularized_nnr import L1RegularizedNNR
from .min_max_scaler import MinMaxScaler as Scaler
from .sklearn import TransformedTargetRegressor, sample_predict

# Show switch to cpu warning
warnings.filterwarnings("default", category=ResourceWarning)

cp = {}

class CheckpointHandler(Callback):
    """CheckpointHandler

    """

    def on_train_end(self, net: L1RegularizedNNR, X=None, y=None, **kwargs):
        print("Loading the best network from the last checkpoint.")
        with io.capture_output() as _:
            net.initialize()
            net.set_input_shape(X, y)
        net.load_params(
            checkpoint=cp['checkpoint'])

class InputShapeSetter(Callback):
    """InputShapeSetter
    dynamically set the input size of the PyTorch module based on the data

    Typically, it’s up to the user to determine the shape of the input data when defining
    the PyTorch module. This can sometimes be inconvenient, e.g. when the shape is only
    known at runtime. E.g., when using :class:`sklearn.feature_selection.VarianceThreshold`,
    you cannot know the number of features in advance. The best solution would be to set
    the input size dynamically.
    """

    def on_train_begin(self, net: L1RegularizedNNR, X=None, y=None, **kwargs):
        net.set_input_shape(X, y)

def _get_nnr(net, callbacks, **l1_regularized_nnr_params):
    return L1RegularizedNNR(
        net,
        callbacks=callbacks,
        **l1_regularized_nnr_params
    )


class TimeSeriesPredictor:
    """Network agnostic time series predictor class

    Parameters
    ----------
    **l1_regularized_nnr_params: skorch L1RegularizedNNR parameters.
    early_stopping: torch.callbacks.EarlyStopping object

    """

    def __init__(self, net, early_stopping: EarlyStopping = None, **l1_regularized_nnr_params):
        self.dataset = None
        self.early_stopping = early_stopping
        self.cpu_count = psutil.cpu_count(logical=True)
        cp['checkpoint'] = Checkpoint(dirname=tempfile.gettempdir())
        if 'train_split' in l1_regularized_nnr_params:
            train_split = l1_regularized_nnr_params.get('train_split')
            if train_split is None:
                cp['checkpoint'].monitor = 'train_loss_best'
                if early_stopping is not None:
                    if early_stopping.monitor == 'valid_loss':
                        raise ValueError(
                            # pylint: disable=line-too-long
                            'Select a valid train_split or disable early_stopping! A valid train_split needs to be selected when valid_loss monitor is selected as early stopping criteria.')
        if 'device' in l1_regularized_nnr_params:
            device = l1_regularized_nnr_params.get('device')
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        l1_regularized_nnr_params['device'] = device

        self.l1_regularized_nnr_params = l1_regularized_nnr_params
        self.ttr = TransformedTargetRegressor(
            regressor=self._get_pipeline(net, cp['checkpoint']),
            transformer=Scaler()
        )

    # # pylint: disable=invalid-name
    # def tune_grid_search(
    #         self,
    #         tunable_params,
    #         dataset: TimeSeriesDataset,
    #         early_stopping=None,
    #         scoring=None,
    #         n_jobs=None,
    #         sk_n_jobs=-1,
    #         cv=5,
    #         refit=True,
    #         verbose=0,
    #         error_score="raise",
    #         return_train_score=False,
    #         local_dir="~/ray_results",
    #         max_iters=1,
    #         use_gpu=False,
    #         **fit_params):
    #     """
    #     More info at https://github.com/ray-project/tune-sklearn/blob/master/examples/torch_nn.py
    #     """
    #     grid_search = TuneGridSearchCV(
    #         self.ttr,
    #         tunable_params,
    #         early_stopping=early_stopping,
    #         scoring=scoring,
    #         n_jobs=n_jobs,
    #         sk_n_jobs=sk_n_jobs,
    #         cv=cv,
    #         refit=refit,
    #         verbose=verbose,
    #         error_score=error_score,
    #         return_train_score=return_train_score,
    #         local_dir=local_dir,
    #         max_iters=max_iters,
    #         use_gpu=use_gpu
    #     )
    #     grid_search.fit(dataset.x, dataset.y, **fit_params)
    #     return (grid_search.best_score_, grid_search.best_params_)

    def _get_pipeline(self, net, _checkpoint):
        return Pipeline(
            [
                ('input scaler', Scaler()),
                ('regressor', _get_nnr(
                        net,
                        [
                            InputShapeSetter(),
                            self.early_stopping,
                            _checkpoint,
                            CheckpointHandler()
                        ],
                        **self.l1_regularized_nnr_params
                    )
                )
            ]
        )

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
        :returns: future dataframe

        """
        x_pred = self.make_future_dataframe(*args, **kwargs)
        return self.sample_predict(x_pred), x_pred

    def forecast(self, *args, **kwargs):
        """Future forecast

        Parameters
        ----------
        *args: variable length unnamed args list
        **kwargs: variable length named args list


        :returns: future forecast
        :returns: future dataframe

        """

        x_pred = self.make_future_dataframe(*args, **kwargs)
        return self.predict(x_pred), x_pred

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
        print(f"Using device {self.l1_regularized_nnr_params.get('device')}")
        self.dataset = dataset
        try:
            self.ttr.fit(dataset.x, dataset.y, **fit_params)
        except RuntimeError as err:
            if str(err).startswith('CUDA out of memory.'):
                warnings.warn(
                    '\nSwitching device to cpu to workaround CUDA out of memory problem.',
                    ResourceWarning)
                self.l1_regularized_nnr_params['device'] = 'cpu'
                self.ttr.regressor = self._get_pipeline(
                    self.ttr.regressor['regressor'].module, cp['checkpoint'])
                self.ttr.fit(dataset.x, dataset.y, **fit_params)
            else:
                raise
        return cp['checkpoint']

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
        score_calculators = [
            ScoreCalculator(
                scores_to_calculate,
                locks,
                self.ttr.score,
                output_list
            ) for _ in range(dataset_length if dataset_length < self.cpu_count else self.cpu_count)
        ]
        for score_calculator in score_calculators:
            score_calculator.start()
        for score_calculator in score_calculators:
            score_calculator.join()
        losses = [loss[1] for loss in output_list]
        # assert all(loss <= 1 for loss in losses)

        device = self.l1_regularized_nnr_params.get('device')
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
