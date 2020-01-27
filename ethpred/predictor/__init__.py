from .clairvoyant_predictor import ClairvoyantPredictor
from .geth_predictor import GethPredictor
from .model_predictor import ModelPredictor


def get(name):
    if name not in globals():
        raise ValueError("predictor {0} does not exist".format(name))
    return globals()[name]
