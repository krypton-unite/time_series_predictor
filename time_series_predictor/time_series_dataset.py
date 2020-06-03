"""
time_series_dataset
"""

import abc
from torch.utils.data import Dataset
from .min_max_scaler import MinMaxScaler

class TimeSeriesDataset(Dataset):
    """
    TimeSeriesDataset

    :param _x: input predictor
    :param _y: output predictor
    """
    def __init__(self, _x, _y, labels):
        super().__init__()
        self.labels = labels
        # Normalize x
        self.scaler_x = MinMaxScaler()
        self._x = self.scaler_x.fit_transform(_x)
        # Normalize y
        self.scaler_y = MinMaxScaler()
        self._y = self.scaler_y.fit_transform(_y)

    def __getitem__(self, idx):
        return (self._x[idx], self._y[idx])

    def __len__(self):
        return self._x.shape[0]

    def get_x_shape(self):
        """get_x_shape

        :returns: input predictor shape
        """
        return self._x.shape

    def get_y_shape(self):
        """get_y_shape

        :returns: output predictor shape
        """
        return self._y.shape

    @abc.abstractmethod
    def make_future_dataframe(self, *args, **kwargs):
        """make_future_dataframe

        :param *args: variable length unnamed args list
        :param **kwargs: variable length named args list
        """
        return
