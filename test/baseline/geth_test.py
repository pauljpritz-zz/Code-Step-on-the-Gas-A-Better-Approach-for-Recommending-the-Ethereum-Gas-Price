import unittest


from ethpred.baseline.geth import GethPredictor


class GethTest(unittest.TestCase):
    def setUp(self):
        raw_prices = [int(v * 10 ** 9) for v in [2, 1, 1, 2, 3]]
        min_prices = {i + 100: v for i, v in enumerate(raw_prices)}
        self.predictor = GethPredictor(min_prices)

    def test_suggest_price(self):
        for percentile, expected in [(60, 2 * 10 ** 9), (100, 3 * 10 ** 9), (20, 10 ** 9)]:
            with self.subTest(percentile=percentile):
                actual = self.predictor.suggest_price(105, blocks_count=1, percentile=percentile)
                self.assertEqual(actual, expected)
