"""
main test script
To run issue the command pytest at the root folder of the project.
"""
from .lstm_tsp import LSTMTimeSeriesPredictor
from .flights_dataset import FlightsDataset

def test_lstm_tsp():
    """
    Tests the LSTMTimeSeriesPredictor
    """
    tsp = LSTMTimeSeriesPredictor(epochs=50)

    tsp.fit(FlightsDataset())
    mean_loss = tsp.compute_mean_loss(tsp.dataloader)
    assert mean_loss < 0.015
