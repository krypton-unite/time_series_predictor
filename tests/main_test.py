"""
main test script
To run, issue the command pytest at the root folder of the project.
"""
import time
from datetime import timedelta

import numpy as np
import pytest
import torch
from flights_time_series_dataset import FlightsDataset
from sklearn.metrics import r2_score  # mean_squared_error
from time_series_models import BenchmarkLSTM, QuantumLSTM
from time_series_predictor import TimeSeriesPredictor


@pytest.mark.skip
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
def test_quantum_lstm_tsp_fitting(device):
    """
    Tests the Quantum LSTM TimeSeriesPredictor fitting
    """
    if device == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip("needs a CUDA compatible GPU available to run this test")
    tsp = TimeSeriesPredictor(
        QuantumLSTM(),
        # lr=1E-2,
        max_epochs=50,
        train_split=None,
        optimizer=torch.optim.Adam,
        device=device
    )

    start = time.time()
    tsp.fit(FlightsDataset())
    end = time.time()
    elapsed = timedelta(seconds = end - start)
    print("Fitting in {} time delta: {}".format(device, elapsed))
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > -3

# @pytest.mark.skip
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
def test_lstm_tsp_forecast(device):
    """
    Tests the LSTMTimeSeriesPredictor forecast
    """
    if device == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip("needs a CUDA compatible GPU available to run this test")
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(hidden_dim=16),
        max_epochs=250,
        train_split=None,
        optimizer=torch.optim.Adam,
        device=device
    )

    whole_fd = FlightsDataset()
    # leave last N months for error assertion
    last_n = 24
    start = time.time()
    tsp.fit(FlightsDataset(last_n))
    end = time.time()
    elapsed = timedelta(seconds = end - start)
    print("Fitting in {} time delta: {}".format(device, elapsed))
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > 0.5

    netout, _ = tsp.forecast(last_n)

    # Select any training example just for comparison
    idx = np.random.randint(0, len(tsp.dataset))
    _, whole_y = whole_fd[idx]

    y_true = whole_y[-last_n:, :]   # get only known future outputs
    y_pred = netout[idx, -last_n:, :]    # get only last N predicted outputs
    r2s = r2_score(y_true, y_pred)
    assert r2s > -1

# @pytest.mark.skip
# def test_lstm_tsp_tuning():
#     """
#     Tests the LSTMTimeSeriesPredictor fitting
#     """
#     if not torch.cuda.is_available():
#         pytest.skip("needs a CUDA compatible GPU available to run this test")
#     tsp = TimeSeriesPredictor(
#         BenchmarkLSTM(hidden_dim=16),
#         max_epochs=50,
#         train_split=None,
#         optimizer=torch.optim.Adam
#     )
#     flights_dataset = FlightsDataset()

#     selected_tunable_params = {
#         "lr": [1e-3, 1e-4]
#     }
#     best_score, best_params = tsp.tune_grid_search(
#         selected_tunable_params,
#         flights_dataset,
#         cv=1,
#         scoring="accuracy",
#         verbose=2,
#         use_gpu=True)
#     print("Best score: {}\tBest params: {}".format(best_score, best_params))
#     tsp.fit(flights_dataset)
#     mean_r2_score = tsp.score(tsp.dataset)
#     assert mean_r2_score > 0.2
