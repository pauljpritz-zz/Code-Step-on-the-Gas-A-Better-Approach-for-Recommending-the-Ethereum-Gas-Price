import numpy as np
import pandas as pd
import torch
from .pipeline.generate_data import generate_data, create_dataloaders
from .models.configure_model import configure_model
from .training.training_loops import GRU_training
from .training.logger import Logger
from .pipeline.generate_data import sliding_window
from .pipeline.to_pandas import convert_to_dataframe
from .pipeline.data_reader import read_data

def prep_data(cnf: dict):
    eth_price, gas_price = read_data(cnf)
    data, _normalizers = convert_to_dataframe(eth_price, gas_price, cnf)
    data.to_pickle(cnf['data']['data_path'])
    print("Data saved to: ", cnf['data']['data_path'])


def run_prepped(cnf: dict):
    np.random.seed(42)
    torch.manual_seed(42)

    # Load stored data
    data = pd.read_pickle(cnf['data']['data_path'])

    if cnf['type'] == 'distribution':
        cnf['data']['y_cols'] = ['mean', 'std_dev']
    y_col_idxs = []
    for col in cnf['data']['y_cols']:
        idx = data.columns.get_loc(col)
        y_col_idxs.append(idx)

    data_df = data.copy()
    data = data.to_numpy()

    X, y = sliding_window(data, cnf['data'])

    y = y[:, :, y_col_idxs]
    y = np.squeeze(y)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Adjust the input size of the model is necessary (needed if all transactions are included)
    cnf['model']['input_size'] = X.shape[2]

    # Split into training and testing data
    data_len = X.shape[0]
    train_len = int(data_len * cnf['data']['train_prop'])
    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]

    # print(X_train[0])
    # print(y_train[0])

    train, test = create_dataloaders(X_train, y_train, X_test, y_test, cnf)
    model = configure_model(cnf)

    logger = Logger(cnf, data=data_df)

    GRU_training(model=model,
                 train_dataloader=train,
                 test_dataloader=test,
                 cnf=cnf['training'],
                 logger=logger)
