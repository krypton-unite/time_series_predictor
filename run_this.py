import numpy as np

from sklearn.metrics import mean_squared_error
from src.flights_dataset import FlightsDataset
from src.lstm_tsp import LSTMTimeSeriesPredictor


if __name__ == "__main__":
    tsp = LSTMTimeSeriesPredictor(epochs=50)

    tsp.fit(FlightsDataset())
    mean_loss = tsp.compute_mean_loss(tsp.dataloader)
