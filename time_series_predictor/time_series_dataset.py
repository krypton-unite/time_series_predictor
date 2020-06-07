"""
time_series_dataset
"""

import abc
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """
    TimeSeriesDataset

    Parameters
    ----------
    param x: input predictor
    param y: output predictor
    """
    def __init__(self, x, y, labels):
        super().__init__()
        self.labels = labels
        self.x = x
        self.y = y

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
