"""
main test script
To run, issue the command pytest at the root folder of the project.
"""
import torch

from src.flights_dataset import FlightsDataset
from src.model import BenchmarkLSTM
from time_series_predictor import TimeSeriesPredictor

if __name__ == "__main__":
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(),
        max_epochs=50,
        train_split=None,
        optimizer=torch.optim.Adam
    )

    tsp.fit(FlightsDataset())
    print(tsp.ttr.regressor_['regressor'].history)
    mean_loss = tsp.score(tsp.dataset)
    assert mean_loss < 2
