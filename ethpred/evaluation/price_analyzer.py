from typing import Dict, Callable
import heapq
import functools
from os import path
import datetime as dt
import json
import logging

from .prediction_stats import PredictionResult, PredictionStats
from ..pipeline.data_reader import read_data
from ..predictor import get as get_predictor


@functools.total_ordering
class TransactionInfo:
    """Information about the the transaction to be included

    Args:
        block_number: the block number at which the transactions has been created
        gas_price: the gas price given for the transaction
    """
    def __init__(self, block_number: int, gas_price: int):
        self.block_number = block_number
        self.gas_price = gas_price

    def __repr__(self):
        return "TransactionInfo(block_number={0}, gas_price={1})".format(
            self.block_number, self.gas_price)

    def __eq__(self, other):
        if not isinstance(other, TransactionInfo):
            return False
        return self.gas_price == other.gas_price

    def __gt__(self, other):
        """We invert the order of gas price to make it nicely work with a min-heap

        >>> TransactionInfo(1, gas_price=2) > TransactionInfo(1, gas_price=3)
        True
        """
        if not isinstance(other, TransactionInfo):
            return NotImplemented
        return self.gas_price < other.gas_price


class MinHeap:
    """Wrapper around Python's heapq
    """

    def __init__(self, values: list = None):
        if values is None:
            values = []
        heapq.heapify(values)
        self.values = values

    def __len__(self):
        return len(self.values)

    def push(self, value):
        """Pushes a value on the heap

        >>> heap = MinHeap([])
        >>> heap.push(1)
        >>> len(heap)
        1
        """
        heapq.heappush(self.values, value)

    def peek(self):
        """Peeks the top value on the heap
        >>> heap = MinHeap([2, 8, 1, 3, 4, 5])
        >>> heap.peek()
        1
        """
        if not self.values:
            return None
        return heapq.nsmallest(1, self.values)[0]

    def pop(self):
        """Pops the top value on the heap
        >>> heap = MinHeap([2, 8, 1, 3, 4, 5])
        >>> heap.pop()
        1
        >>> len(heap)
        5
        """
        if not self.values:
            return None
        return heapq.heappop(self.values)


class PriceAnalyzer:
    """Analyze the gas price difference and the average time for inclusion

    Args:
        min_prices: A mapping of block height to minimum price
        predict_price: predict_price should be a function which takes
                       a block height and returns a gas price prediction/suggestion
    """
    def __init__(self, min_prices: Dict[int, int], predict_price: Callable[[int], int]):
        self.min_prices = min_prices
        self.predict_price = predict_price

    def analyze_prices(self, start_block: int, end_block: int, final_block: int) -> PredictionStats:
        """Returns stats about gas price difference and the average time for inclusion

        Args:
            start_block: block at which to start the computations (inclusive)
            end_block: block at which to end the computations (inclusive)
            final_block: block at which we should give up on including transactions
                         in case one was underpriced and never got included
        """
        waiting_for_inclusion = MinHeap()
        stats = PredictionStats()
        block_number = start_block

        total_blocks_count = final_block - start_block

        # wait until we added transactions until end_blocks
        # and included all of them or reached the final block after which we give up
        while block_number <= end_block or \
              (waiting_for_inclusion and block_number <= final_block):
            if block_number % 100 == 0:
                logging.info("progress: %s/%s", block_number - start_block, total_blocks_count)
            current_price = self.min_prices.get(block_number)
            if not current_price:
                block_number += 1
                continue

            # include all possible transactions waiting for inclusion
            # in practice the price might not be enough for all the transactions
            # to be included but that will be for version 2.0
            while waiting_for_inclusion and waiting_for_inclusion.peek().gas_price >= current_price:
                transaction = waiting_for_inclusion.pop()
                stats.add(PredictionResult(transaction, block_number, current_price))

            # if we already went above the end_block we are just waiting for
            # transactions to be included
            if block_number <= end_block:
                predicted_price = self.predict_price(block_number)
                waiting_for_inclusion.push(TransactionInfo(block_number, predicted_price))

            block_number += 1

        # add stats about transactions which could not be included
        while waiting_for_inclusion:
            transaction = waiting_for_inclusion.pop()
            stats.add(PredictionResult(transaction))

        return stats


def make_filename(basename):
    filename, extname = path.splitext(basename)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = "{0}-{1}{2}".format(filename, timestamp, extname)
    return output_file


def run_analysis(cnf):
    logging.info("loading data")
    _eth_price, gas_price = read_data(cnf)
    logging.info("data loaded")

    start_block = gas_price[-1]["block_number"] + cnf["evaluation"].get("skip", {}).get("start", 0)
    final_block = gas_price[0]["block_number"]
    end_block = final_block - cnf["evaluation"].get("skip", {}).get("end", 0)
    min_prices = {v["block_number"]: v["min_price_tx"]["gas_price"] for v in gas_price}

    Predictor = get_predictor(cnf["evaluation"]["predictor"]["class"])
    predictor = Predictor.from_cnf(gas_price, cnf["evaluation"]["predictor"]["args"], cnf)

    price_analyzer = PriceAnalyzer(min_prices, predictor.predict_price)

    logging.info("running evaluation")
    stats = price_analyzer.analyze_prices(start_block, end_block, final_block)
    output_file = make_filename(cnf["evaluation"]["output_basename"])

    output = dict(
        stats=stats.to_dict(),
        cnf=cnf,
    )
    with open(output_file, "w") as f:
        json.dump(output, f)
