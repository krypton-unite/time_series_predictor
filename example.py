"""
Tests the LSTMTimeSeriesPredictor
"""
from pathlib import Path

import torch
from skorch.callbacks import EarlyStopping
from src.flights_dataset import FlightsDataset
from src.model import BenchmarkLSTM
from src.oze_dataset import OzeNPZDataset
from oze_dataset import npz_check
from time_series_predictor import TimeSeriesPredictor
from dotenv import load_dotenv
import os
# from tune_sklearn.tune_gridsearch import TuneGridSearchCV

if __name__ == "__main__":
    load_dotenv(Path('src', '.env.test.local'))
    credentials = {}
    credentials['user_name'] = os.getenv("CHALLENGE_USER_NAME")
    credentials['user_password'] = os.getenv("CHALLENGE_USER_PASSWORD")
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(),
        max_epochs=500,
        early_stopping=EarlyStopping(patience=30),
        # train_split=None, # default = skorch.dataset.CVSplit(5)
        optimizer=torch.optim.Adam
    )
    dataset = OzeNPZDataset(
        dataset_path=npz_check(
            Path('docs', 'source', 'notebooks', 'datasets'),
            'dataset',
            credentials=credentials
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
