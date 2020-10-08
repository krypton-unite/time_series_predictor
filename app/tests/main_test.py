"""
main test script
To run, issue the command pytest at the root folder of the project.
"""
import numpy as np
import torch
import pytest
from sklearn.metrics import r2_score  # mean_squared_error

from app.src.flights_dataset import FlightsDataset
from app.src.model import BenchmarkLSTM
from app.time_series_predictor import TimeSeriesPredictor

def test_lstm_tsp_fitting():
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA compatible GPU available to run this test")
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(hidden_dim=16),
        max_epochs=50,
        train_split=None,
        optimizer=torch.optim.Adam
    )

    tsp.fit(FlightsDataset())
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > 0.2

def test_lstm_tsp_fitting_in_cpu():
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(hidden_dim=16),
        max_epochs=50,
        train_split=None,
        optimizer=torch.optim.Adam,
        device='cpu'
    )

    tsp.fit(FlightsDataset())
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > 0.2

def test_lstm_tsp_forecast():
    """
    Tests the LSTMTimeSeriesPredictor forecast
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA compatible GPU available to run this test")
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(hidden_dim=16),
        max_epochs=1000,
        train_split=None,
        optimizer=torch.optim.Adam
    )

    whole_fd = FlightsDataset()
    # leave last N months for error assertion
    last_n = 24
    tsp.fit(FlightsDataset(last_n))
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > 0.98

    netout = tsp.sample_forecast(last_n)

    # Select any training example just for comparison
    idx = np.random.randint(0, len(tsp.dataset))
    _, whole_y = whole_fd[idx]

    y_true = whole_y[-last_n:, :]   # get only known future outputs
    y_pred = netout[-last_n:, :]    # get only last N predicted outputs
    r2s = r2_score(y_true, y_pred)
    assert r2s > -1

def test_lstm_tsp_forecast_in_cpu():
    """
    Tests the LSTMTimeSeriesPredictor forecast
    """
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(hidden_dim=16),
        max_epochs=1000,
        train_split=None,
        optimizer=torch.optim.Adam,
        device='cpu'
    )

    whole_fd = FlightsDataset()
    # leave last N months for error assertion
    last_n = 24
    tsp.fit(FlightsDataset(last_n))
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > 0.98

    netout = tsp.sample_forecast(last_n)

    # Select any training example just for comparison
    idx = np.random.randint(0, len(tsp.dataset))
    _, whole_y = whole_fd[idx]

    y_true = whole_y[-last_n:, :]   # get only known future outputs
    y_pred = netout[-last_n:, :]    # get only last N predicted outputs
    r2s = r2_score(y_true, y_pred)
    assert r2s > -1
