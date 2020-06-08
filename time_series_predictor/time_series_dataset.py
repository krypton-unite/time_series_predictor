"""
time_series_dataset
"""

import abc
from torch.utils.data import Dataset
from .three_d_min_max_scaler import ThreeDMinMaxScaler as Scaler
# from .min_max_scaler import MinMaxScaler as Scaler

class TimeSeriesDataset(Dataset):
    """
    TimeSeriesDataset

    :param _x: input predictor
    :param _y: output predictor
    """
    # pylint: disable=invalid-names
    def __init__(self, _x, _y, labels):
        super().__init__()
        self.labels = labels
        # Normalize x
        # self.scaler_x = Scaler()
        # self.x = self.scaler_x.fit_transform(_x)
        self.x = _x
        # Normalize y
        self.scaler_y = Scaler()
        self.y = self.scaler_y.fit_transform(_y)
        self.y = _y

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

    def __len__(self):
        return self.x.shape[0]

    def get_x_shape(self):
        """get_x_shape

        :returns: input predictor shape
        """
        return self.x.shape

    def get_y_shape(self):
        """get_y_shape

        :returns: output predictor shape
        """
        return self.y.shape

    @abc.abstractmethod
    def make_future_dataframe(self, *args, **kwargs):
        """make_future_dataframe

        Parameters
        ----------
        *args: variable length unnamed args list
        **kwargs: variable length named args list
        """
