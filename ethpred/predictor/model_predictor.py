from typing import List
import datetime as dt
import bisect

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from .predictor import Predictor
from ..pipeline.inference import predict_prices


class ModelPredictor(Predictor):
    """ModelPredictor uses a pre-trained model information to predict the
    ideal gas cost
    """
    def __init__(self, gas_price: List[dict],
                 timestamps: List[dt.datetime],
                 predictions: List[np.ndarray],
                 percentile: int = 20,
                 utility: float = 0.9):
        super().__init__(gas_price)
        self.timestamps = timestamps
        self.predictions = predictions
        self.percentile = percentile
        self.utility = utility
        self.trends = self._compute_trends()

    def _find_predictions(self, timestamp):
        index = bisect.bisect_right(self.timestamps, timestamp) - 1
        return self.predictions[index], self.trends[index]

    @classmethod
    def from_cnf(cls, min_prices: List[dict], kwargs: dict, cnf: dict):
        model = torch.load(kwargs['model_path'])
        timestamps, predictions = predict_prices(cnf, model)
        return cls(min_prices, timestamps, predictions,
                   kwargs.get('percentile', 20), kwargs.get('utility', 0.9))

    def get_block_datetime(self, block_number: int) -> int:
        datetime = self.get_datetime(block_number)
        while not datetime:
            block_number -= 1
            datetime = self.get_datetime(block_number)
        return datetime

    def _compute_trends(self):
        trends = []
        for pred in self.predictions:
            trend = self.get_trend(pred)
            trends.append(trend)
        scaler = MinMaxScaler((-2, 0))
        trends = scaler.fit_transform(np.array(trends).reshape(-1, 1)).reshape(-1)
        return trends

    def get_trend(self, predictions):
        model = LinearRegression()
        X = np.arange(len(predictions)).reshape(-1, 1)
        model.fit(X, predictions)
        return model.coef_[0]

    def predict_price(self, block_number: int) -> int:
        datetime = self.get_block_datetime(block_number)
        predictions, trend = self._find_predictions(datetime)
        value = np.percentile(predictions, q=self.percentile)
        coefficient = np.exp(trend) * self.utility
        return value * coefficient
