# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Example usage
# ## Imports

# %%
import torch
import numpy as np
from matplotlib import pyplot as plt

from src.lstm_tsp import LSTMTimeSeriesPredictor
from src.flights_dataset import FlightsDataset

# %% [markdown]
# ## Config

# %%
plot_config = {}
plot_config['training progress'] = True
plot_config['prediction on training data'] = True
plot_config['forecast'] = True

forecast_config = {}
forecast_config['include history'] = True
forecast_config['months ahead'] = 24

predictor_config = {}
predictor_config['epochs'] = 1000
predictor_config['learning rate'] = 1e-2
predictor_config['hidden dim'] = 100
predictor_config['layers num'] = 3

config = {}
config['plot'] = plot_config
config['forecast'] = forecast_config
config['predictor'] = predictor_config
config['predict on training data enabled'] = True
config['forecast enabled'] = True

# %% [markdown]
# ## Time Series Predictor instantiation

# %%
tsp = LSTMTimeSeriesPredictor(
    hidden_dim=config['predictor']['hidden dim'],
    num_layers=config['predictor']['layers num'],
    lr=config['predictor']['learning rate'],
    max_epochs=config['predictor']['epochs'],
    train_split=None,
    optimizer=torch.optim.Adam                # torch.optim.SGD
)

# %% [markdown]
# ## Training process

# %%
ds = FlightsDataset()
tsp.fit(ds)
# training_dataframe = tsp.get_training_dataframe()

if config['plot']['training progress']:
    history_length = len(tsp.neural_net_regressor.history)
    train_loss = np.zeros((history_length, 1))
    for epoch in tsp.neural_net_regressor.history:
        epoch_number = epoch['epoch']-1
        train_loss[epoch_number] = epoch['train_loss']
    plt.figure(figsize=(20, 20))
    plt.plot(train_loss, 'o-', label='training')
    plt.axes().set_xlabel('Epoch')
    plt.axes().set_ylabel('MSE')
    plt.legend()

# %% [markdown]
# ## Prediction on training data

# %%
if config['predict on training data enabled']:
    # Select training example
    idx = np.random.randint(0, len(tsp.dataset))
    dataloader = tsp.neural_net_regressor.get_iterator(tsp.dataset)
    x, y = dataloader.dataset[idx]

    # Run predictions
    netout = tsp.predict(x)

    d_output = netout.shape[1]
    for idx_output_var in range(d_output):
        # Select real passengers data
        y_true = y[:, idx_output_var]

        y_pred = netout[:, idx_output_var]

        if config['plot']['prediction on training data']:
            plt.figure(figsize=(20, 20))
            plt.subplot(d_output, 1, idx_output_var+1)

            plt.plot(y_true, label="Truth")
            plt.plot(y_pred, label="Prediction")
            plt.title(tsp.dataset.labels['y'][idx_output_var])
            plt.legend()

# %% [markdown]
# ## Future forecast

# %%
# Run forecast
if config['forecast enabled']:
    netout = tsp.forecast(config['forecast']['months ahead'],
                          include_history=config['forecast']['include history'])

    d_output = netout.shape[1]
    # Select any training example just for comparison
    idx = np.random.randint(0, len(tsp.dataset))
    dataloader = tsp.neural_net_regressor.get_iterator(tsp.dataset)
    x, y = dataloader.dataset[idx]
    for idx_output_var in range(d_output):
        # Select real passengers data
        y_true = y[:, idx_output_var]

        y_pred = netout[:, idx_output_var]

        if config['plot']['forecast']:
            plt.figure(figsize=(20, 20))
            plt.subplot(d_output, 1, idx_output_var+1)

            if config['forecast']['include history']:
                plot_args = [y_pred]
            else:
                y_pred_index = [i+tsp.dataset.get_x_shape()[1]+1 for i in range(len(y_pred))]
                plot_args = [y_pred_index, y_pred]
            plt.plot(*plot_args, label="Prediction")
            plt.plot(y_true, label="Truth")
            plt.title(tsp.dataset.labels['y'][idx_output_var])
            plt.legend()
