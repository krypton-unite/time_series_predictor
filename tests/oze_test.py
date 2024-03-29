"""early_stopping_test
"""
from pathlib import Path

import pytest
import torch
from flights_time_series_dataset import FlightsDataset
from oze_dataset import OzeNPZDataset, npz_check
from skorch.callbacks import EarlyStopping
from time_series_models import BenchmarkLSTM
from time_series_predictor import TimeSeriesPredictor
from .config import devices
from .helpers import cuda_check

def _get_credentials(user_name, user_password):
    credentials = {}
    credentials['user_name'] = user_name
    credentials['user_password'] = user_password
    return credentials

def _get_dataset(user_name, user_password):
    return OzeNPZDataset(
        dataset_path=npz_check(
            Path('docs', 'source', 'notebooks', 'datasets'),
            'dataset',
            credentials=_get_credentials(user_name, user_password)
        )
    )

@pytest.mark.skip
@pytest.mark.parametrize('device', devices)
def test_regular(user_name, user_password, device):
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    cuda_check(device)
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(hidden_dim=16),
        early_stopping=EarlyStopping(patience=30),
        max_epochs=500,
        # train_split=None, # default = skorch.dataset.CVSplit(5)
        optimizer=torch.optim.Adam,
        device=device
    )

    tsp.fit(_get_dataset(user_name, user_password))
    mean_r2_score = tsp.score(tsp.dataset)
    print(f"Mean R2 score achieved: {mean_r2_score}")
    assert mean_r2_score > -100

@pytest.mark.skip
@pytest.mark.parametrize('device', devices)
def test_no_train_split(device):
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    cuda_check(device)
    with pytest.raises(ValueError) as error:
        TimeSeriesPredictor(
            BenchmarkLSTM(hidden_dim=16),
            early_stopping=EarlyStopping(),
            max_epochs=500,
            train_split=None,
            optimizer=torch.optim.Adam,
            device=device
        )
    # pylint: disable=line-too-long
    assert error.match(
        'Select a valid train_split or disable early_stopping! A valid train_split needs to be selected when valid_loss monitor is selected as early stopping criteria.'
    )

@pytest.mark.skip
@pytest.mark.parametrize('device', devices)
def test_train_loss_monitor_no_train_split(device):
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    cuda_check(device)
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(hidden_dim=10),
        early_stopping=EarlyStopping(monitor='train_loss', patience=15),
        max_epochs=150,
        train_split=None,
        optimizer=torch.optim.Adam,
        device=device
    )
    tsp.fit(FlightsDataset())
    mean_r2_score = tsp.score(tsp.dataset)
    print(f"Mean R2 score achieved: {mean_r2_score}")
    assert mean_r2_score > -300

@pytest.mark.skip
@pytest.mark.parametrize('device', devices)
def test_train_loss_monitor(user_name, user_password, device):
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    cuda_check(device)
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(hidden_dim=10),
        early_stopping=EarlyStopping(monitor='train_loss', patience=15),
        max_epochs=150,
        # train_split=None, # default = skorch.dataset.CVSplit(5)
        optimizer=torch.optim.Adam,
        device=device
    )
    tsp.fit(_get_dataset(user_name, user_password))
    mean_r2_score = tsp.score(tsp.dataset)
    print(f"Mean R2 score achieved: {mean_r2_score}")
    assert mean_r2_score > -300
