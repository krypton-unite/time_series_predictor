"""
main test script
To run, issue the command pytest at the root folder of the project.
"""
import torch
from src.flights_dataset import FlightsDataset
from src.lstm_tsp import LSTMTimeSeriesPredictor


if __name__ == "__main__":
    tsp = LSTMTimeSeriesPredictor(max_epochs=50, train_split=None, optimizer=torch.optim.Adam)

    tsp.fit(FlightsDataset())
    mean_loss = tsp.score(tsp.dataset)
    assert mean_loss < 2
