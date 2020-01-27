from typing import Tuple, List
import datetime as dt

import numpy as np
import torch

from .data_reader import read_data
from .to_pandas import convert_to_dataframe
from .generate_data import sliding_window



def predict_prices(cnf, model) -> Tuple[List[dt.datetime], List[np.ndarray]]:
    eth_price, gas_price = read_data(cnf)
    data, normalizers = convert_to_dataframe(eth_price, gas_price, cnf)
    data_array = data.to_numpy()
    X, _y, indices = sliding_window(data_array, cnf['data'], return_indices=True)
    timestamps = [v.to_pydatetime() for v in data.iloc[indices].index.tolist()]
    X = torch.from_numpy(X).float()
    with torch.no_grad():
        normalized_predictions = model(X).numpy()
    # FIXME: only works when predicting a single series
    normalizer = normalizers[cnf['data']['y_cols'][0]]
    predictions = normalizer.inverse_transform(normalized_predictions)
    return timestamps, predictions
