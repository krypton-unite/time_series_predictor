"""
Tests the LSTMTimeSeriesPredictor
"""
from pathlib import Path

import torch
from skorch.callbacks import EarlyStopping
from app.src.flights_dataset import FlightsDataset
from app.src.model import BenchmarkLSTM
from app.src.oze_dataset import OzeNPZDataset, npz_check
from app.time_series_predictor import TimeSeriesPredictor
# from tune_sklearn.tune_gridsearch import TuneGridSearchCV

if __name__ == "__main__":
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(),
        max_epochs=500,
        early_stopping=EarlyStopping(patience=30),
        # train_split=None, # default = skorch.dataset.CVSplit(5)
        optimizer=torch.optim.Adam
    )
    dataset = OzeNPZDataset(
        dataset_path=npz_check(
            Path('app', 'datasets'),
            'dataset'
        )
    )

    tsp.fit(dataset)
    mean_r2_score = tsp.score(dataset)
    assert mean_r2_score > -50

# if __name__ == "__main__":
#     tsp = TimeSeriesPredictor(
#         BenchmarkLSTM(),
#         max_epochs=50,
#         train_split=None, # default = skorch.dataset.CVSplit(5)
#         optimizer=torch.optim.Adam
#     )
#     dataset = FlightsDataset()

#     tsp.fit(dataset)
#     mean_r2_score = tsp.score(dataset)
#     assert mean_r2_score > 0.75
