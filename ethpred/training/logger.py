import os
from os import path
import time
import json
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import torch
import yaml


class Logger:
    def __init__(self, cnf):
        self.cnf = cnf
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H_%M_%S.%f")
        self.save_path = path.join(cnf['training']['log_path'],
                                   'model_' + str(self.timestamp) + '/')

    def plot_loss_hist(self, hist_train, hist_test):
        plt.figure(figsize=(14, 8))
        plt.plot(hist_train, color='b', label='train')
        plt.plot(hist_test, color='r', label='test')
        plt.legend()
        plt.savefig(self.save_path + 'training_hist.png')
        if self.cnf['training']['show_plots']:
            plt.show()

    def plot_test_prediction_example(self, y_true, y_pred):
        plt.figure(figsize=(14, 8))
        plt.plot(y_true, color='b', label='True')
        plt.plot(y_pred, color='r', label='Predicted')
        plt.legend()
        plt.savefig(self.save_path + 'test_prediction_example.png')
        if self.cnf['training']['show_plots']:
            plt.show()

    def dump_results(self, model, hist_train, hist_test, res_dict):
        os.mkdir(self.save_path)
        self.plot_loss_hist(hist_train, hist_test)

        with open(self.save_path + 'config_and_results.txt', 'w') as f:
            print("########CONFIG########", file=f)
            f.write(yaml.dump(self.cnf, default_flow_style=False))
            print("########RESULTS########", file=f)
            f.write(yaml.dump(res_dict, default_flow_style=False))

        if self.cnf['training']['save_model']:
            torch.save(model, self.save_path + 'model.pickle')
