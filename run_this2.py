"""
run_this.py
"""
from pathlib import Path

import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate
import skorch
from skorch import NeuralNetRegressor

from src.flights_dataset import FlightsDataset
from src.lstm_tsp import LSTMTimeSeriesPredictor
from src.model import BenchmarkLSTM
from src.oze_dataset import OzeNPZDataset, npz_check

if __name__ == "__main__":
    plot_config = {}
    plot_config['training progress'] = False
    plot_config['prediction on training data'] = False
    plot_config['forecast'] = True

    forecast_config = {}
    forecast_config['include history'] = True
    forecast_config['months ahead'] = 24

    predictor_config = {}
    predictor_config['epochs'] = 100
    predictor_config['learning rate'] = 1e-2
    predictor_config['hidden dim'] = 100
    predictor_config['layers num'] = 3

    config = {}
    config['plot'] = plot_config
    config['forecast'] = forecast_config
    config['predictor'] = predictor_config
    config['predict enabled'] = False
    config['forecast enabled'] = True

    tsp = LSTMTimeSeriesPredictor(
        hidden_dim=config['predictor']['hidden dim'],
        num_layers=config['predictor']['layers num'],
        lr=config['predictor']['learning rate'],  # 0.01
        max_epochs=5, # config['predictor']['epochs'], # 10
        train_split=None,                         # skorch.dataset.CVSplit(5)
        optimizer=torch.optim.Adam                # torch.optim.SGD
    )
    ds = FlightsDataset()
    t0 = time.perf_counter()
    tsp.fit(ds)
    print(time.perf_counter()-t0, 'seconds')
    history_length = len(tsp.neural_net_regressor.history)
    train_loss = np.zeros((history_length, 1))
    for epoch in tsp.neural_net_regressor.history:
        epoch_number = epoch['epoch']-1
        train_loss[epoch_number] = epoch['train_loss']
    plt.plot(train_loss, 'o-', label='training')
    plt.legend()
    plt.show()

    # tsp = LSTMTimeSeriesPredictor(
    #     hidden_dim=config['predictor']['hidden dim'],
    #     num_layers=config['predictor']['layers num'],
    #     lr=config['predictor']['learning rate'],
    #     max_epochs=5, # max_epochs=config['predictor']['epochs'],
    #     train_split=skorch.dataset.CVSplit(5),
    #     optimizer=torch.optim.Adam                # torch.optim.SGD
    # )
    # ds = OzeNPZDataset(
    #     dataset_path=npz_check(Path('datasets'), 'dataset')
    # )
    # t0 = time.perf_counter()
    # tsp.fit(ds)
    # print(time.perf_counter()-t0, 'seconds')
    # history_length = len(tsp.neural_net_regressor.history)
    # train_loss = np.zeros((history_length, 1))
    # valid_loss = np.zeros((history_length, 1))
    # for epoch in tsp.neural_net_regressor.history:
    #     epoch_number = epoch['epoch']-1
    #     train_loss[epoch_number] = epoch['train_loss']
    #     valid_loss[epoch_number] = epoch['valid_loss']
    # plt.plot(train_loss, 'o-', label='training')
    # plt.plot(valid_loss, 'o-', label='validation')
    # plt.legend()
    # plt.show()

    # plt.plot(hist_loss, 'o-', label='train')
    # plt.legend()
    # plt.show()
    # # training_dataframe = tsp.get_training_dataframe()

    # ds = OzeNPZDataset(
    #     dataset_path=npz_check(Path('datasets'), 'dataset')
    # )

    # d_input = ds.get_x_shape()[2]     # From dataset
    # d_output = ds.get_y_shape()[2]    # From dataset
    # net = BenchmarkLSTM(input_dim=d_input, hidden_dim=config['predictor']['hidden dim'],
    #                     output_dim=d_output, num_layers=config['predictor']['layers num'])

    # net_regr = NeuralNetRegressor(
    #     net,
    #     max_epochs=config['predictor']['epochs'],
    #     lr=config['predictor']['learning rate'],
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    # )

    # net_regr.fit(ds.x, ds.y)
    pass
