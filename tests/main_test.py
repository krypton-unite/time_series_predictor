"""
main test script
To run, issue the command pytest at the root folder of the project.
"""
import numpy as np

import torch
from sklearn.metrics import r2_score # mean_squared_error
from time_series_predictor import TimeSeriesPredictor
from src.model import BenchmarkLSTM
from src.flights_dataset import FlightsDataset


def test_lstm_tsp_fitting():
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(),
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
        BenchmarkLSTM(),
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
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(),
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
