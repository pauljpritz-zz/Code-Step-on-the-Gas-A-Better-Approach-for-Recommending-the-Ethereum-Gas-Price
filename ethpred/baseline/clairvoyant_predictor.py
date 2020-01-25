from typing import Dict


class ClairvoyantPredictor:
    """The ClairvoyantPredictor cheats and uses prices from the future
    to predict prices in the future. More precisely, it uses the lowest
    price up to the given number of blocks to predict the price value

    This is used to evaluate what is the maximum savings that a contract
    could do by waiting for at most X blocks before submitting the transaction

    NOTE: currently very unotmipized: looks at `blocks_to_wait` blocks every time
          to find the minimum. Should be easy enough with a heap + dict
          pointing to the heap but unless we use a crazy number of blocks it
          should be fast enough anyway

    Args:
        min_prices: A mapping of block height to minimum price
        blocks_to_wait: The maximum number of blocks to wait for the transactions
        to be included
    """
    def __init__(self, min_prices: Dict[int, int], blocks_to_wait: int = 240):
        self.prices = min_prices
        self.blocks_to_wait = blocks_to_wait

    def predict_price(self, block_number: int) -> int:
        """Returns the minimal gas price in the next ``blocks_to_wait``

        Args:
            block_number: Block number for which the price should be predicted.
        Returns: gas price
        """
        start_block = block_number + 1
        end_block = start_block + self.blocks_to_wait
        return min(self.prices[i] for i in range(start_block, end_block))
