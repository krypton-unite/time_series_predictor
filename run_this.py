"""
Tests the LSTMTimeSeriesPredictor
"""
from pathlib import Path

import torch
from src.flights_dataset import FlightsDataset
from src.model import BenchmarkLSTM
from src.oze_dataset import OzeNPZDataset, npz_check
from time_series_predictor import TimeSeriesPredictor

# if __name__ == "__main__":
#     tsp = TimeSeriesPredictor(
#         BenchmarkLSTM(),
#         max_epochs=5,
#         # train_split=None, # default = skorch.dataset.CVSplit(5)
#         optimizer=torch.optim.Adam
#     )
#     dataset = OzeNPZDataset(
#         dataset_path=npz_check(
#             Path('datasets'),
#             'dataset'
#         )
#     )

#     tsp.fit(dataset)
#     mean_r2_score = tsp.score(dataset)
#     assert mean_r2_score > -300

if __name__ == "__main__":
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(),
        max_epochs=50,
        train_split=None, # default = skorch.dataset.CVSplit(5)
        optimizer=torch.optim.Adam
    )
    dataset = FlightsDataset()

    tsp.fit(dataset)
    mean_r2_score = tsp.score(dataset)
    assert mean_r2_score > 0.75
