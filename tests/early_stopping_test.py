"""early_stopping_test
"""
import time
from datetime import timedelta

import pytest
import torch
from torch.optim import Adam
from flights_time_series_dataset import FlightSeriesDataset
from skorch.callbacks import EarlyStopping
from skorch.dataset import CVSplit
from time_series_models import BenchmarkLSTM
from time_series_predictor import TimeSeriesPredictor


# @pytest.mark.skip
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
def test_regular(device):
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    if device == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip("needs a CUDA compatible GPU available to run this test")

    start = time.time()
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(hidden_dim=16),
        lr = 1e-3,
        lambda1=1e-8,
        optimizer__weight_decay=1e-8,
        iterator_train__shuffle=True,
        early_stopping=EarlyStopping(patience=50),
        max_epochs=250,
        train_split=CVSplit(10),
        optimizer=Adam,
        device=device,
    )

    past_pattern_length = 24
    future_pattern_length = 12
    pattern_length = past_pattern_length + future_pattern_length
    fsd = FlightSeriesDataset(pattern_length, past_pattern_length, pattern_length)
    tsp.fit(fsd)
    end = time.time()
    elapsed = timedelta(seconds = end - start)
    print("Fitting in {} time delta: {}".format(device, elapsed))
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > -100

# @pytest.mark.skip
def test_no_train_split():
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    with pytest.raises(ValueError) as error:
        TimeSeriesPredictor(
            BenchmarkLSTM(hidden_dim=16),
            early_stopping=EarlyStopping(),
            train_split=None
        )
    # pylint: disable=line-too-long
    assert error.match(
        'Select a valid train_split or disable early_stopping! A valid train_split needs to be selected when valid_loss monitor is selected as early stopping criteria.'
    )

# @pytest.mark.skip
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('train_split', [None, CVSplit(5)])
def test_train_loss_monitor(device, train_split):
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    if device == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip("needs a CUDA compatible GPU available to run this test")

    start = time.time()
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(hidden_dim=16),
        lr = 1e-3,
        lambda1=1e-8,
        optimizer__weight_decay=1e-8,
        iterator_train__shuffle=True,
        early_stopping=EarlyStopping(monitor='train_loss', patience=50),
        max_epochs=250,
        train_split=train_split,
        optimizer=Adam,
        device=device,
    )

    past_pattern_length = 24
    future_pattern_length = 12
    pattern_length = past_pattern_length + future_pattern_length
    fsd = FlightSeriesDataset(pattern_length, past_pattern_length, pattern_length)
    tsp.fit(fsd)
    end = time.time()
    elapsed = timedelta(seconds = end - start)
    print("Fitting in {} time delta: {}".format(device, elapsed))
    mean_r2_score = tsp.score(tsp.dataset)
    print("Mean R2 Score: {}".format(mean_r2_score))
    assert mean_r2_score > -100
