from typing import List

import numpy as np

from .predictor import Predictor


MAX_PRICE: int = 500_000_000_000


class GethPredictor(Predictor):
    """Geth 'algorithm' to suggest a price is very straight forward
    It simply uses a tunable percentile for the minimum price in the past blocks
    The percentile and number of look-behind blocks can be set when starting geth

    Original logic can be found here:
    https://github.com/ethereum/go-ethereum/blob/master/eth/gasprice/gasprice.go

    Args:
        gas_price: A list of gas prices information about blocks
        blocks_count: This number will be multiplied by 5 to obtain
                      the number of blocks to look behind and divided by 2
                      to find the number of empty blocks to skip
            percentile: This is the percentile to use when suggesting the price
    """
    def __init__(self,
                 gas_price: List[dict],
                 blocks_count: int = 20,
                 percentile: float = 60,
                 factor: float = 1.0):
        super().__init__(gas_price)
        self.blocks_count = blocks_count
        self.percentile = min(max(0, percentile), 100)
        self.max_blocks = self.blocks_count * 5
        self.factor = factor

    @classmethod
    def from_cnf(cls, min_prices: List[dict], kwargs: dict, _cnf: dict):
        return cls(min_prices, **kwargs)

    def predict_price(self, block_number: int) -> int:
        """Mimics the logic of geth to suggest a gas price

        Args:
            block_number: Block number for which the price should be predicted.
                          Block must be in the min_prices give during initialization
        Returns: gas price
        """
        max_empty = self.blocks_count // 2
        prices = []
        current_block_number = block_number - 1
        while block_number >= 0 and len(prices) < self.max_blocks:
            price = self.get_min_price(current_block_number, 0)
            if price == 0 and max_empty >= 0:
                max_empty -= 1
                continue
            prices.append(price)
            current_block_number -= 1
        suggested_price = int(np.percentile(prices, q=self.percentile))
        if suggested_price >= MAX_PRICE:
            suggested_price = MAX_PRICE
        return suggested_price * self.factor
