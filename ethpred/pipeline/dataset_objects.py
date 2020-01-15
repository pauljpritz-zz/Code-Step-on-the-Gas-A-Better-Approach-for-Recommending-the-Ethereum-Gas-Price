import torch
import torch.utils.data as du

class TimeSeriesData(du.Dataset):
    """
    Wrapper for time series data to be used by pytorch DataLoader.
    """

    def __init__(self, X, y, batch_first=True):
        self.batch_first = batch_first
        self.X = X
        self.y = y

    def __getitem__(self, item):
        if self.batch_first:
            x_t = self.X[item]
            y_t = self.y[item]
            return x_t, y_t
        else:
            x_t = self.X[:, item]
            y_t = self.y[:, item]
            return x_t, y_t

    def __len__(self):
        if self.batch_first:
            return self.X.shape[0]
        else:
            return self.X.shape[1]