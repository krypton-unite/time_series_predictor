"""
time_series_predictor script
"""
import datetime
import warnings

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .time_series_dataset import TimeSeriesDataset

# Show switch to cpu warning
warnings.filterwarnings("default", category=ResourceWarning)

# pylint: disable=too-many-instance-attributes
class TimeSeriesPredictor:
    """Network agnostic time series predictor class

    :param learning_rate:
    :param epochs:
    """
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cpu_count = psutil.cpu_count(logical=False)
        self.device = None
        self.net = None
        self.dataloader = None
        self.dataset = None
        self.loss_function = None
        self.optimizer = None
        self.model_save_path = f'model_{datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")}.pth'

    def _config_fit(self, dataset: TimeSeriesDataset, device):
        batch_size = 100
        self.device = device
        is_cuda = device == torch.device("cuda:0")
        num_workers = 0 if is_cuda else self.cpu_count
        # More info about memory pinning here:
        # https://pytorch.org/docs/stable/data.html#memory-pinning
        self.dataloader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     pin_memory=is_cuda,
                                     num_workers=num_workers)
        self.net = self.net.to(device)
        self.dataset = dataset

    def _fit(self):
        loss_best = np.inf
        # Prepare loss history
        hist_loss = np.zeros(self.epochs)
        for idx_epoch in range(self.epochs):
            running_loss = 0
            with tqdm(total=len(self.dataloader.dataset),
                      desc=f"[Epoch {idx_epoch+1:3d}/{self.epochs}]") as pbar:
                for idx_batch, (inp, out) in enumerate(self.dataloader):
                    self.optimizer.zero_grad()

                    # Propagate input
                    net_out = self.net(inp.to(self.device))

                    # Compute loss
                    loss = self.loss_function(out.to(self.device), net_out)

                    # Backpropagate loss
                    loss.backward()

                    # Update weights
                    self.optimizer.step()

                    running_loss += loss.item()
                    pbar.set_postfix({'loss': running_loss/(idx_batch+1)})
                    pbar.update(inp.shape[0])

                train_loss = running_loss/len(self.dataloader)
                pbar.set_postfix({'loss': train_loss})

                hist_loss[idx_epoch] = train_loss

                if train_loss < loss_best:
                    loss_best = train_loss
                    best_net = self.net
        if best_net:
            torch.save(best_net.state_dict(), self.model_save_path)
            print(f"\nmodel exported to {self.model_save_path} with loss {loss_best:5f}")
            self.net = best_net
        return hist_loss

    def get_training_dataframe(self):
        """get_training_dataframe

        :returns: training dataframe
        """
        return [inp for (inp, out) in self.dataloader]

    def make_future_dataframe(self, *args, **kwargs):
        # pylint: disable=anomalous-backslash-in-string
        """make_future_dataframe

        :param \*args: variable length unnamed args list
        :param \*\*kwargs: variable length named args list
        :returns: future dataframe
        """
        return self.dataset.make_future_dataframe(*args, **kwargs)

    def forecast(self, *args, **kwargs):
        # pylint: disable=anomalous-backslash-in-string
        """Future forecast

        :param \*args: variable length unnamed args list
        :param \*\*kwargs: variable length named args list
        """
        return self.predict(self.make_future_dataframe(*args, **kwargs))

    def predict(self, inp):
        """Run predictions

        :param inp: input
        """
        with torch.no_grad():
            return self.net(torch.Tensor(inp[np.newaxis, :, :]).to(self.device)).cpu().numpy()

    def fit(self, dataset: TimeSeriesDataset, net, loss_function=torch.nn.MSELoss()):
        """Fit selected network

        :param dataset: dataset to fit on
        :param net: network to use
        :param loss_function: optional loss function to use
        """
        self.net = net
        self._config_fit(dataset, torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"))
        print(f"Using device {self.device}")
        self.loss_function = loss_function
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        try:
            return self._fit()
        except RuntimeError as err:
            if str(err).startswith('CUDA out of memory.'):
                warnings.warn(
                    '\nSwitching device to cpu to workaround CUDA out of memory problem.',
                    ResourceWarning)
                self._config_fit(dataset, torch.device("cpu"))
                return self._fit()
            raise

    def compute_loss(self, dataloader):
        """Compute the loss of a network on a given dataset.

        Does not compute gradient.

        :param dataloader: iterator on the dataset.
        :returns: loss with no grad.
        """
        dataloader_length = len(dataloader)
        loss = np.empty(dataloader_length)
        with torch.no_grad():
            for idx_batch, (inp, out) in enumerate(dataloader):
                net_out = self.net(inp.to(self.device))
                loss[idx_batch] = self.loss_function(out.to(self.device), net_out)

        return loss

    def compute_mean_loss(self, dataloader):
        """Compute the mean loss of a network on a given dataset.

        Does not compute gradient.

        :param dataloader: iterator on the dataset.
        :returns: mean loss with no grad.
        """
        return np.mean(self.compute_loss(dataloader))
