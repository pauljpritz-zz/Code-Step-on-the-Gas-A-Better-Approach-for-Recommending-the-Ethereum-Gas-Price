import datetime as dt


class Predictor:
    def __init__(self, gas_price):
        self.raw_prices = gas_price
        self.indexed_prices = {v["block_number"]: v for v in gas_price}

    def get_min_price(self, block_number: int, default: int = None) -> int:
        block = self.indexed_prices.get(block_number)
        if not block:
            return default
        return block["min_price_tx"]["gas_price"]

    def get_datetime(self, block_number: int) -> dt.datetime:
        block = self.indexed_prices.get(block_number)
        if not block:
            return None
        return dt.datetime.fromtimestamp(block["timestamp"])

    def predict_price(self, block_number: int) -> int:
        raise NotImplementedError()
