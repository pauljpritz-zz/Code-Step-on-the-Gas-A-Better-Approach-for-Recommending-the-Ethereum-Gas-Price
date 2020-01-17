import torch
import torch.nn as nn


class DistributionGRU(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 input_size: int,
                 num_layers: int,
                 pred_steps: int,
                 dropout: int,
                 linear_units: int):
        super(DistributionGRU, self).__init__()
        self.GRU = nn.GRU(batch_first=True,
                          input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout)
        # Introduce two dims to forecast both mean and std. dev.
        self.pred_steps = pred_steps
        self.num_pred_dims = 2
        self.linearLayer_1 = nn.Linear(in_features=hidden_size, out_features=linear_units)
        self.act_func = nn.Sigmoid()
        self.linearLayer_2 = nn.Linear(in_features=linear_units,
                                       out_features=pred_steps * self.num_pred_dims)

    def forward(self, X):
        # TODO: Check what exact input shape we need here and if we need to expand it.
        encoded = self.GRU(X)
        # Use the last hidden state of the GRU
        pred = self.linearLayer_1(encoded[0][:, 0])
        pred = self.act_func(pred)
        pred = self.linearLayer_2(pred)
        # Return two dimensions, one for mean one for std. dev.
        pred = pred.reshape([pred.shape[0], self.pred_steps, -1])
        return pred


def configure_DistributionGRU(cnf):
    return DistributionGRU(hidden_size=cnf['hidden_size'],
                           input_size=cnf['input_size'],
                           num_layers=cnf["num_layers"],
                           pred_steps=cnf['pred_steps'],
                           linear_units=cnf['linear_units'],
                           dropout=cnf['dropout'])
