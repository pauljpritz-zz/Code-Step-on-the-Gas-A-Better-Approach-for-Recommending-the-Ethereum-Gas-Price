from typing import List
import datetime as dt
import bisect

import numpy as np
import torch

from .predictor import Predictor
from ..pipeline.inference import predict_prices


class ModelPredictor(Predictor):
    """ModelPredictor uses a pre-trained model information to predict the
    ideal gas cost
    """
    def __init__(self, gas_price: List[dict],
                 timestamps: List[dt.datetime],
                 predictions: List[np.ndarray],
                 percentile: int = 20):
        super().__init__(gas_price)
        self.timestamps = timestamps
        self.predictions = predictions
        self.percentile = percentile

    def _find_predictions(self, timestamp):
        index = bisect.bisect_right(self.timestamps, timestamp) - 1
        return self.predictions[index]

    @classmethod
    def from_cnf(cls, min_prices: List[dict], kwargs: dict, cnf: dict):
        model = torch.load(kwargs['model_path'])
        timestamps, predictions = predict_prices(cnf, model)
        return cls(min_prices, timestamps, predictions, kwargs.get('percentile', 20))

    def get_block_datetime(self, block_number: int) -> int:
        datetime = self.get_datetime(block_number)
        while not datetime:
            block_number -= 1
            datetime = self.get_datetime(block_number)
        return datetime

    def predict_price(self, block_number: int) -> int:
        datetime = self.get_block_datetime(block_number)
        predictions = self._find_predictions(datetime)
        return np.percentile(predictions, q=self.percentile)
