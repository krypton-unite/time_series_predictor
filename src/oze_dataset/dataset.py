# coding: UTF-8
"""This module defines an example Torch dataset from the Oze datachallenge.

Example
-------
$ dataloader = DataLoader(OzeDataset(DATSET_PATH),
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)

"""

import json
from pathlib import Path

import numpy as np

from time_series_predictor import TimeSeriesDataset

TIME_SERIES_LENGTH = 672

class OzeNPZDataset(TimeSeriesDataset):
    """Torch dataset for Oze datachallenge evaluation.

    Load dataset from a single npz file.

    Attributes
    ----------
    x: np.array
        Dataset input of shape (m, K, 37).
    y: np.array
        Dataset target of shape (m, K, 8).
    labels: Dict
        Ordered labels list for R, Z and X.
    m: np.array
        Normalization constant.
    M: np.array
        Normalization constant.
    """

    def __init__(self,
                 dataset_path,
                 labels_path=Path(__file__).parent.joinpath('labels.json')):
        """Load dataset from npz.

        Parameters
        ---------
        dataset_x: str or Path
            Path to the dataset inputs as npz.
        labels_path: str or Path, optional
            Path to the labels, divided in R, Z and X, in json format.
            Default is "labels.json".
        """
        def load_npz(dataset_path, labels_path):
            """Load dataset from csv and create x_train and y_train tensors."""
            def make_predictor(features):
                #pylint: disable=too-many-function-args
                return np.concatenate(features, axis=-1).astype(np.float32)
            # Load dataset as csv
            dataset = np.load(dataset_path)

            # Load labels, can be found through csv or challenge description
            with open(labels_path, "r") as stream_json:
                labels = json.load(stream_json)

            R, X, Z = dataset['R'], dataset['X'], dataset['Z']
            m = Z.shape[0]  # Number of training examples
            K = Z.shape[-1]  # Time serie length

            R = np.tile(R[:, np.newaxis, :], (1, K, 1))
            Z = Z.transpose((0, 2, 1))
            X = X.transpose((0, 2, 1))

            # Store input features
            input_features = [Z, R]
            x = make_predictor(input_features)

            # Store output features
            output_features = [X]
            y = make_predictor(output_features)

            return (x, y, labels)
        super().__init__(*load_npz(dataset_path, labels_path))

    def make_future_dataframe(self, *args, **kwargs):
        pass
