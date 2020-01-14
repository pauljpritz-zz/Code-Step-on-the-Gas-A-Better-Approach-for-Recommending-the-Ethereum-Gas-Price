import torch
import torch.nn as nn

class SimpleGRU(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 input_size: int,
                 num_layers: int,
                 pred_steps: int,
                 dropout: int,
                 linear_units: int):
        super(SimpleGRU, self).__init__()
        self.GRU = nn.GRU(batch_first=True,
                          input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout)
        self.linearLayer_1 = nn.Linear(in_features=hidden_size, out_features=linear_units)
        self.act_func = nn.Sigmoid()
        self.linearLayer_2 = nn.Linear(in_features=linear_units, out_features=pred_steps)

    def forward(self, X):
        # TODO: Check what exact input shape we need here and if we need to expand it.
        encoded = self.GRU_layer(X)
        # Use the last hidden state of the GRU
        pred = self.linearLayer_1(encoded[0][:, 0])
        pred = self.act_func(pred)
        pred = self.linearLayer_2(pred)
        return pred