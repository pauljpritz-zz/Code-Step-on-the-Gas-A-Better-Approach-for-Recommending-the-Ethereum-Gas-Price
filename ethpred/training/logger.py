import os
from os import path
import copy
import time
import json
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import torch
import yaml

from ..pipeline.generate_data import generate_data
from ..pipeline.generate_data import sliding_window

class Logger:
    def __init__(self, cnf, data=None):
        self.cnf = cnf
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H_%M_%S.%f")
        self.save_path = path.join(cnf['training']['log_path'],
                                   'model_' + str(self.timestamp) + '/')
        self.data = data

    def plot_loss_hist(self, hist_train, hist_test):
        plt.figure(figsize=(14, 8))
        plt.plot(hist_train, color='b', label='train')
        plt.plot(hist_test, color='r', label='test')
        plt.legend()
        plt.savefig(self.save_path + 'training_hist.png')
        if self.cnf['training']['show_plots']:
            plt.show()

    def dump_results(self, model, hist_train, hist_test, res_dict):
        os.mkdir(self.save_path)
        self.plot_loss_hist(hist_train, hist_test)
        self.generate_prediction_example(model)

        with open(self.save_path + 'config_and_results.txt', 'w') as f:
            print("########CONFIG########", file=f)
            f.write(yaml.dump(self.cnf, default_flow_style=False))
            print("########RESULTS########", file=f)
            f.write(yaml.dump(res_dict, default_flow_style=False))

        if self.cnf['training']['save_model']:
            torch.save(model, self.save_path + 'model.pickle')

    def generate_prediction_example(self, model):
        gen_cnf = copy.deepcopy(self.cnf)
        gen_cnf['data']['sample_freq'] = gen_cnf['data']['y_len']


        if self.data is not None:
            if gen_cnf['type'] == 'distribution':
                gen_cnf['data']['y_cols'] = ['mean', 'std_dev']
            y_col_idxs = []
            for col in gen_cnf['data']['y_cols']:
                idx = self.data.columns.get_loc(col)
                y_col_idxs.append(idx)

            data = self.data.to_numpy()

            X, y = sliding_window(data, gen_cnf['data'])

            y = y[:, :, y_col_idxs]
            y = np.squeeze(y)

            print("X shape:", X.shape)
            print("y shape:", y.shape)

            # Adjust the input size of the model is necessary (needed if all transactions are included)
            gen_cnf['model']['input_size'] = X.shape[2]

            # Split into training and testing data
            data_len = X.shape[0]
            train_len = int(data_len * gen_cnf['data']['train_prop'])
            X_train, y_train = X[:train_len], y[:train_len]
            X_test, y_test = X[train_len:], y[train_len:]


        else:
            X_train, X_test, y_train, y_test = generate_data(gen_cnf)

        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()

        with torch.no_grad():
            y_pred = model(X_test)
            y_pred_train = model(X_train)

        plt.figure(figsize=(14, 8))
        if gen_cnf['type'] == 'distribution':
            y_pred = y_pred.reshape([-1, 2])
            y_test = y_test.reshape([-1, 2])
            mean_pred = y_pred[:, 0]
            std_pred = y_pred[:, 1]
            mean = y_test[:, 0]
            std = y_test[:, 1]

            x = np.arange(mean.shape[0])
            plt.plot(x, mean, color='b', label='True')
            plt.fill_between(x, mean-1.96*std, mean+1.96*std, color='royalblue', alpha=0.5)
            plt.plot(x, mean_pred, color='r', label='Predicted')
            plt.fill_between(x, mean_pred-1.96*std_pred, mean_pred+1.96*std_pred, color='tomato', alpha=0.5)
        else:
            y_pred_train = y_pred_train.reshape(-1)
            y_train = y_train.reshape(-1)

            y_pred = y_pred.reshape(-1)
            y_test = y_test.reshape(-1)

            y_pred = torch.cat((y_pred_train, y_pred))
            y_test = torch.cat((y_train, y_test))

            x = np.arange(y_pred.shape[0])
            plt.plot(x, y_test, color='b', label='True')
            plt.plot(x, y_pred, color='r', label='Predicted')

        plt.legend()
        plt.savefig(self.save_path + 'test_prediction_example.png')
        if self.cnf['training']['show_plots']:
            plt.show()
