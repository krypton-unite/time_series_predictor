"""
main test script
To run, issue the command pytest at the root folder of the project.
"""
import numpy as np

import torch
from sklearn.metrics import mean_squared_error
from src.flights_dataset import FlightsDataset
from src.lstm_tsp import LSTMTimeSeriesPredictor


def test_lstm_tsp_fitting():
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    tsp = LSTMTimeSeriesPredictor(max_epochs=50, train_split=None, optimizer=torch.optim.Adam)

    tsp.fit(FlightsDataset())
    mean_loss = tsp.compute_mean_loss(tsp.pipe['regressor'].get_iterator(tsp.dataset))
    assert mean_loss < 0.015 # 140000

def test_lstm_tsp_forecast():
    """
    Tests the LSTMTimeSeriesPredictor forecast
    """
    tsp = LSTMTimeSeriesPredictor(max_epochs=1000, train_split=None, optimizer=torch.optim.Adam)

    whole_fd = FlightsDataset()
    # leave last N months for error assertion
    last_n = 24
    tsp.fit(FlightsDataset(last_n))
    mean_loss = tsp.compute_mean_loss(tsp.pipe['regressor'].get_iterator(tsp.dataset))
    assert mean_loss < 0.001 # 14000

    netout = tsp.forecast(last_n)
    d_output = netout.shape[1]

    # Select any training example just for comparison
    idx = np.random.randint(0, len(tsp.dataset))
    _, whole_y = whole_fd[idx]
    for idx_output_var in range(d_output):
        # Select real passengers data
        y_true = whole_y[-last_n:, idx_output_var]      # get only known future outputs
        y_pred = netout[-last_n:, idx_output_var]  # get only last N predicted outputs
        assert mean_squared_error(y_true, y_pred) < 0.02 # 90000

# def test_lstm_tsp_get_training_dataframe():
#     """
#     Tests the LSTMTimeSeriesPredictor get_training_dataframe
#     """
#     tsp = LSTMTimeSeriesPredictor(max_epochs=50)

#     tsp.fit(FlightsDataset())
#     training_df = tsp.get_training_dataframe()

#     idx = np.random.randint(0, len(tsp.dataset))
#     inp_df, _ = tsp.dataloader.dataset[idx]
#     assert np.array_equal(training_df[0].cpu().numpy()[idx, :, :], inp_df)
