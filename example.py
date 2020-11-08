"""
Tests the LSTMTimeSeriesPredictor
"""
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from flights_time_series_dataset import FlightsDataset
from oze_dataset import OzeNPZDataset, npz_check
from skorch.callbacks import EarlyStopping

from time_series_models import BenchmarkLSTM
from time_series_predictor import TimeSeriesPredictor

# from tune_sklearn.tune_gridsearch import TuneGridSearchCV

# if __name__ == "__main__":
#     load_dotenv(Path('test', '.env.test.local'))
#     credentials = {
#         'user_name': os.getenv("CHALLENGE_USER_NAME"),
#         'user_password': os.getenv("CHALLENGE_USER_PASSWORD")
#     }
#     tsp = TimeSeriesPredictor(
#         BenchmarkLSTM(),
#         max_epochs=500,
#         early_stopping=EarlyStopping(patience=30),
#         # train_split=None, # default = skorch.dataset.CVSplit(5)
#         optimizer=torch.optim.Adam
#     )
#     dataset = OzeNPZDataset(
#         dataset_path=npz_check(
#             Path('docs', 'source', 'notebooks', 'datasets'),
#             'dataset',
#             credentials=credentials
#         )
#     )

#     tsp.fit(dataset)
#     mean_r2_score = tsp.score(dataset)
#     assert mean_r2_score > -50

if __name__ == "__main__":
    tsp = TimeSeriesPredictor(
        # BenchmarkLSTM(hidden_dim=3000),
        BenchmarkLSTM(),
        max_epochs=50,
        train_split=None, # default = skorch.dataset.CVSplit(5)
        optimizer=torch.optim.Adam,
        # device='cpu'
    )
    dataset = FlightsDataset()

    tsp.fit(dataset)
    mean_r2_score = tsp.score(dataset)
    print(mean_r2_score)
    assert mean_r2_score > 0.75
