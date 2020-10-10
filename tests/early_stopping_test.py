"""early_stopping_test
"""
from pathlib import Path

import pytest
import torch
from skorch.callbacks import EarlyStopping

from src.flights_dataset import FlightsDataset
from src.model import BenchmarkLSTM
from src.oze_dataset import OzeNPZDataset, npz_check
from time_series_predictor import TimeSeriesPredictor


def _get_credentials(user_name, user_password):
    credentials = {}
    credentials['user_name'] = user_name
    credentials['user_password'] = user_password
    return credentials

def _get_dataset(user_name, user_password):
    return OzeNPZDataset(
        dataset_path=npz_check(
            Path('datasets'),
            'dataset',
            credentials=_get_credentials(user_name, user_password)
        )
    )

def test_regular(user_name, user_password):
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(hidden_dim=16),
        early_stopping=EarlyStopping(patience=30),
        max_epochs=500,
        # train_split=None, # default = skorch.dataset.CVSplit(5)
        optimizer=torch.optim.Adam
    )

    tsp.fit(_get_dataset(user_name, user_password))
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > -100

def test_no_train_split():
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    with pytest.raises(ValueError) as error:
        TimeSeriesPredictor(
            BenchmarkLSTM(hidden_dim=16),
            early_stopping=EarlyStopping(),
            max_epochs=500,
            train_split=None,
            optimizer=torch.optim.Adam
        )
    # pylint: disable=line-too-long
    assert error.match(
        'Select a valid train_split or disable early_stopping! A valid train_split needs to be selected when valid_loss monitor is selected as early stopping criteria.'
    )

def test_train_loss_monitor_no_train_split():
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(hidden_dim=10),
        early_stopping=EarlyStopping(monitor='train_loss', patience=15),
        max_epochs=150,
        train_split=None,
        optimizer=torch.optim.Adam
    )
    tsp.fit(FlightsDataset())
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > -300

def test_train_loss_monitor(user_name, user_password):
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(hidden_dim=10),
        early_stopping=EarlyStopping(monitor='train_loss', patience=15),
        max_epochs=150,
        # train_split=None, # default = skorch.dataset.CVSplit(5)
        optimizer=torch.optim.Adam
    )
    tsp.fit(_get_dataset(user_name, user_password))
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > -300
