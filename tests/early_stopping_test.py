"""early_stopping_test
"""
from pathlib import Path

import torch
from skorch.callbacks import EarlyStopping

from src.flights_dataset import FlightsDataset
from src.model import BenchmarkLSTM
from src.oze_dataset import OzeNPZDataset, npz_check
from time_series_predictor import TimeSeriesPredictor

def _get_dataset():
    return OzeNPZDataset(
        dataset_path=npz_check(
            Path('datasets'),
            'dataset'
        )
    )

def test_regular():
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(),
        early_stopping=EarlyStopping(patience=15),
        max_epochs=500,
        # train_split=None, # default = skorch.dataset.CVSplit(5)
        optimizer=torch.optim.Adam
    )

    tsp.fit(_get_dataset())
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > -200

def test_no_train_split():
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    try:
        TimeSeriesPredictor(
            BenchmarkLSTM(),
            early_stopping=EarlyStopping(),
            max_epochs=500,
            train_split=None,
            optimizer=torch.optim.Adam
        )
        raise Exception("Should have raised an exception!")
    # pylint: disable=broad-except
    except Exception as exception:
        assert type(exception).__name__ == 'ValueError'
        # pylint: disable=line-too-long
        assert str(exception) == 'Select a valid train_split or disable early_stopping! A valid train_split needs to be selected when valid_loss monitor is selected as early stopping criteria.'

def test_train_loss_monitor_no_train_split():
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(),
        early_stopping=EarlyStopping(monitor='train_loss', patience=30),
        max_epochs=500,
        train_split=None,
        optimizer=torch.optim.Adam
    )
    tsp.fit(FlightsDataset())
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > 0.2

def test_train_loss_monitor():
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(),
        early_stopping=EarlyStopping(monitor='train_loss', patience=30),
        max_epochs=500,
        # train_split=None, # default = skorch.dataset.CVSplit(5)
        optimizer=torch.optim.Adam
    )
    tsp.fit(_get_dataset())
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > 0.2
