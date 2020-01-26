from .clairvoyant_predictor import ClairvoyantPredictor
from .geth_predictor import GethPredictor


def get(name):
    if name not in globals():
        raise ValueError("predictor {0} does not exist".format(name))
    return globals()[name]
