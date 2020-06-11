"""
main test script
To run, issue the command pytest at the root folder of the project.
"""
import numpy as np

import torch
from sklearn.metrics import mean_squared_error
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
    mean_loss = tsp.score(tsp.dataset)
    assert mean_loss < 1.7

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
    mean_loss = tsp.score(tsp.dataset)
    assert mean_loss < 1.7

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
    mean_loss = tsp.score(tsp.dataset)
    assert mean_loss < 2

    netout = tsp.sample_forecast(last_n)
    d_output = netout.shape[-1]

    # Select any training example just for comparison
    idx = np.random.randint(0, len(tsp.dataset))
    _, whole_y = whole_fd[idx]
    for idx_output_var in range(d_output):
        # Select real passengers data
        y_true = whole_y[-last_n:, idx_output_var]      # get only known future outputs
        y_pred = netout[-last_n:, idx_output_var]  # get only last N predicted outputs
        mse = mean_squared_error(y_true, y_pred)
        assert mse < 10000

# def test_lstm_tsp_get_training_dataframe():
#     """
#     Tests the LSTMTimeSeriesPredictor get_training_dataframe
#     """
#     tsp = TimeSeriesPredictor(
#         BenchmarkLSTM(),
#         max_epochs=50,
#         train_split=None,
#         optimizer=torch.optim.Adam
#     )

#     tsp.fit(FlightsDataset())
#     training_df = tsp.get_training_dataframe()

#     idx = np.random.randint(0, len(tsp.dataset))
#     inp_df, _ = tsp.dataloader.dataset[idx]
#     assert np.array_equal(training_df[0].cpu().numpy()[idx, :, :], inp_df)
