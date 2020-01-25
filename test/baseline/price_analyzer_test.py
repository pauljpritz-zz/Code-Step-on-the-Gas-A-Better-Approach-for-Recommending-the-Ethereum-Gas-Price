import unittest

from ethpred.baseline.price_analyzer import PriceAnalyzer


class PriceAnalyzerTest(unittest.TestCase):
    def test_analyze_prices(self):
        raw_prices = [10, 9, 5, 6, 8, 7, 10]
        prices = dict(enumerate(raw_prices))
        predictions = {0: 6, 1: 5, 2: 4, 3: 7, 4: 9, 5: 8}
        predict_price = lambda block_number: predictions[block_number]
        price_analyzer = PriceAnalyzer(prices, predict_price)
        stats = price_analyzer.analyze_prices(0, 5, 6)
        self.assertEqual(len(stats), 6)
        # transactions at block 2 and 5 have not been included
        not_included = [v for v in stats if not v["included"]]
        self.assertEqual(len(not_included), 2)
        self.assertEqual(not_included[0]["creation_gas_price"], 8)
        self.assertEqual(not_included[1]["creation_gas_price"], 4)

        included = [v for v in stats if v["included"]]
        self.assertEqual(len(included), 4)
        self.assertEqual(included[0]["creation_gas_price"], 6)
        self.assertEqual(included[0]["gas_price_diff"], 1)
        self.assertEqual(included[0]["blocks_waited"], 2)
        self.assertEqual(included[1]["creation_gas_price"], 5)
        self.assertEqual(included[1]["gas_price_diff"], 0)
        self.assertEqual(included[1]["blocks_waited"], 1)
