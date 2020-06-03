"""
main test script
To run, issue the command pytest at the root folder of the project.
"""
import numpy as np

from sklearn.metrics import mean_squared_error
from src.flights_dataset import FlightsDataset
from src.lstm_tsp import LSTMTimeSeriesPredictor


def test_lstm_tsp_fitting():
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    tsp = LSTMTimeSeriesPredictor(epochs=50)

    tsp.fit(FlightsDataset())
    mean_loss = tsp.compute_mean_loss(tsp.dataloader)
    assert mean_loss < 0.015

def test_lstm_tsp_forecast():
    """
    Tests the LSTMTimeSeriesPredictor forecast
    """
    tsp = LSTMTimeSeriesPredictor(epochs=2000)

    whole_fd = FlightsDataset()
    # leave last N months for error assertion
    last_n = 24
    tsp.fit(FlightsDataset(last_n))
    mean_loss = tsp.compute_mean_loss(tsp.dataloader)
    assert mean_loss < 0.001

    netout = tsp.forecast(last_n)
    d_output = netout.shape[2]

    # Select any training example just for comparison
    idx = np.random.randint(0, len(tsp.dataloader.dataset))
    _, whole_y = whole_fd[idx]
    for idx_output_var in range(d_output):
        # Select real passengers data
        y_true = whole_y[-last_n:, idx_output_var]      # get only known future outputs
        y_pred = netout[idx, -last_n:, idx_output_var]  # get only last N predicted outputs
        assert mean_squared_error(y_true, y_pred) < 0.02
