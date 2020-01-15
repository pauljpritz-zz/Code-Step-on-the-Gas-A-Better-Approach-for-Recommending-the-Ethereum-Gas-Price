import numpy as np
import pandas as pd
from .data_reader import read_data
from .to_pandas import convert_to_dataframe


def generate_data(cnf: dict):
    eth_price, gas_price = read_data(cnf)
    data = convert_to_dataframe(eth_price, gas_price, cnf['data'])
    data = data.to_numpy()

    # TODO: Transform to numpy here and return dataset loaders. us some sliding window stuff.

    print(data.shape)


def sliding_window(data: np.ndarray, cnf_data: dict):
    window_size = cnf_data['window_size']
    y_len = cnf_data['y_len']
    sample_freq = cnf_data['sample_freq']

    overlap = (data.shape[0] - window_size) % (y_len)
    print("truncated by:", overlap)
    if overlap != 0:
        data = data[overlap:]

    X_idx_start = np.arange((data.shape[0] - window_size - y_len) // (sample_freq + 1))
    X_idx = np.arange(y_len)[None, :] + X_idx_start[:, None]
    print(X_idx.shape)
    X = data[X_idx]
    return X
