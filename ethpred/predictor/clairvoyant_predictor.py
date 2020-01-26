from typing import Dict
import math

from sortedcontainers import SortedList


# any big value should do, it is just so that it is not returned by min
NOT_FOUND_GAS = 1e20


class ClairvoyantPredictor:
    """The ClairvoyantPredictor cheats and uses prices from the future
    to predict prices in the future. More precisely, it uses the lowest
    price up to the given number of blocks to predict the price value

    This is used to evaluate what is the maximum savings that a contract
    could do by waiting for at most X blocks before submitting the transaction

    Args:
        min_prices: A mapping of block height to minimum price
        blocks_to_wait: The maximum number of blocks to wait for the transactions
        to be included
        max_stdevs: the maximum number of standard deviations under which the gas price
                    is not allowed to be. This is used to avoid using a total
                    outlier gas price
    """
    def __init__(self, min_prices: Dict[int, int], blocks_to_wait: int = 240,
                 max_stdevs: float = None):
        self.prices = min_prices
        self.blocks_to_wait = blocks_to_wait
        self.max_stdevs = max_stdevs
        self._available_prices = SortedList()
        self._initialized = False

        # to compute mean and stdev in O(1)
        self._rolling_price_total = 0
        self._rolling_squared_total = 0

    @classmethod
    def from_cnf(cls, min_prices: Dict[int, int], kwargs: dict):
        return cls(min_prices, **kwargs)

    def current_price_mean(self):
        return self._rolling_price_total / len(self._available_prices)

    def current_price_stdev(self):
        mean = self.current_price_mean()
        second_moment = self._rolling_squared_total / len(self._available_prices)
        variance = second_moment - mean ** 2
        return math.sqrt(variance)

    def _add_price(self, block_number):
        price = self.prices.get(block_number)
        if price:
            self._rolling_price_total += price
            self._rolling_squared_total += price * price
            self._available_prices.add(price)

    def _remove_price(self, block_number):
        price = self.prices.get(block_number)
        if price:
            try:
                self._available_prices.remove(price)
                self._rolling_price_total -= price
                self._rolling_squared_total -= price * price
            except ValueError:
                pass

    def _initialize(self, start_block, end_block):
        for block_number in range(start_block, end_block + 1):
            self._add_price(block_number)
        self._initialized = True

    def predict_price(self, block_number: int) -> int:
        """Returns the minimal gas price in the next ``blocks_to_wait``

        Args:
            block_number: Block number for which the price should be predicted.
        Returns: gas price
        """
        end_block = block_number + self.blocks_to_wait
        if not self._initialized:
            self._initialize(block_number + 1, end_block)
        else:
            self._remove_price(block_number)
            self._add_price(end_block)
        if self.max_stdevs is None:
            return self._available_prices[0]
        min_price = self.current_price_mean() - self.max_stdevs * self.current_price_stdev()
        for price in self._available_prices:
            if price >= min_price:
                return price
        raise ValueError("no possible price found")
