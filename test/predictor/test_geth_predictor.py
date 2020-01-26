import unittest


from ethpred.predictor.geth_predictor import GethPredictor


class GethPredictorTest(unittest.TestCase):
    def setUp(self):
        raw_prices = [int(v * 10 ** 9) for v in [2, 1, 1, 2, 3]]
        self.min_prices = {i + 100: v for i, v in enumerate(raw_prices)}

    def test_predict_price(self):
        for percentile, expected in [(60, 2 * 10 ** 9), (100, 3 * 10 ** 9), (20, 10 ** 9)]:
            with self.subTest(percentile=percentile):
                predictor = GethPredictor(self.min_prices, blocks_count=1, percentile=percentile)
                actual = predictor.predict_price(105)
                self.assertEqual(actual, expected)
