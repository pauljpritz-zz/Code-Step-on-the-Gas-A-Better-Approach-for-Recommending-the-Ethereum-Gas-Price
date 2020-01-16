import numpy as np
import pandas as pd
import torch.utils.data as du
import torch
from .data_reader import read_data
from .to_pandas import convert_to_dataframe
from .dataset_objects import TimeSeriesData
from .calc_distributions import generate_distribution


def generate_data(cnf: dict):
    eth_price, gas_price = read_data(cnf)
    data = convert_to_dataframe(eth_price, gas_price, cnf)

    # Only include the columns wanted for y
    if cnf['type'] == 'distribution':
        cnf['data']['y_cols'] = ['mean', 'std_dev']
    y_col_idxs = []
    for col in cnf['data']['y_cols']:
        idx = data.columns.get_loc(col)
        y_col_idxs.append(idx)

    data = data.to_numpy()

    X, y = sliding_window(data, cnf['data'])
    y = y[:, :, y_col_idxs]
    y = np.squeeze(y)


    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Adjust the input size of the model is necessary (needed if all transactions are included
    cnf['model']['input_size'] = X.shape[2]

    # Split into training and testing data
    data_len = X.shape[0]
    train_len = int(data_len * cnf['data']['train_prop'])
    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]

    # Create dataloaders
    train_dataloader = create_dataloader(X_train, y_train, cnf['data']['batch_size'])
    test_dataloader = create_dataloader(X_test, y_test, cnf['data']['batch_size'])
    return train_dataloader, test_dataloader


def sliding_window(data: np.ndarray, cnf_data: dict):
    window_size = cnf_data['window_size']
    y_len = cnf_data['y_len']
    sample_freq = cnf_data['sample_freq']

    overlap = (data.shape[0] - window_size) % (y_len)
    print("truncated by:", overlap)
    if overlap != 0:
        data = data[overlap:]

    X_idx_start = sample_freq * np.arange(
        (data.shape[0] - window_size - y_len) // (sample_freq + 1))
    X_idx = np.arange(window_size)[None, :] + X_idx_start[:, None]
    X = data[X_idx]

    y_idx_start = X_idx_start + window_size
    y_idx = np.arange(y_len)[None, :] + y_idx_start[:, None]
    y = data[y_idx]
    return X, y


def create_dataloader(X, y, batch_size):
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    data = TimeSeriesData(X, y)
    dataloader = du.DataLoader(dataset=data, batch_size=batch_size)
    return dataloader
