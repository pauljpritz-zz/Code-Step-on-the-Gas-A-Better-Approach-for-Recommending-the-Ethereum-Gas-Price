import unittest

from ethpred.baseline.clairvoyant_predictor import ClairvoyantPredictor


class ClairvoyantPredictorTest(unittest.TestCase):
    def test_predict_price(self):
        prices = dict(enumerate([5, 7, 8, 5, 6, 2, 4]))
        predictor = ClairvoyantPredictor(prices, 4)
        self.assertEqual(predictor.predict_price(0), 5)
        self.assertEqual(predictor.predict_price(1), 2)
        self.assertEqual(predictor.predict_price(2), 2)
