from typing import List

import numpy as np


class PredictionResult:
    """Contains the transaction block number and gas price as well
    as the inclusion block and gas price if the transaction was included
    """
    def __init__(self, transaction, inclusion_block_number=None, inclusion_gas_price=None):
        self.transaction = transaction
        self.inclusion_block_number = inclusion_block_number
        self.inclusion_gas_price = inclusion_gas_price

    @property
    def included(self):
        return self.inclusion_block_number is not None

    @property
    def blocks_waited(self):
        if not self.included:
            raise ValueError("transaction has not been included")
        return self.inclusion_block_number - self.transaction.block_number

    @property
    def gas_price_diff(self):
        if not self.included:
            raise ValueError("transaction has not been included")
        return self.transaction.gas_price - self.inclusion_gas_price

    def to_dict(self):
        result = dict(
            included=self.included,
            creation_block_number=self.transaction.block_number,
            creation_gas_price=self.transaction.gas_price,
        )
        if self.included:
            result.update(dict(
                inclusion_block_number=self.inclusion_block_number,
                inclusion_gas_price=self.inclusion_gas_price,
                blocks_waited=self.blocks_waited,
                gas_price_diff=self.gas_price_diff,
            ))
        return result

    def __repr__(self):
        return "PredictionResult(transaction={0}, included={1})".format(
            self.transaction, self.included)


class PredictionStats:
    """PredictionStats is a container to store the results of the evaluation
    """
    def __init__(self):
        self.stats: List[PredictionResult] = []

    def add(self, result: PredictionResult):
        self.stats.append(result)

    def __len__(self):
        return len(self.stats)

    def __iter__(self):
        return iter(self.stats)

    def compute_total_gas_price_diff(self):
        return sum(v.gas_price_diff for v in self.stats if v.included)

    def compute_total_blocks_waited(self):
        return sum(v.blocks_waited for v in self.stats if v.included)

    def compute_included_count(self):
        return sum(1 for v in self.stats if v.included)

    def compute_average_gas_price(self):
        prices = [v.transaction.gas_price for v in self.stats if v.included]
        return sum(prices) / len(prices)

    def compute_median_gas_price(self):
        prices = [v.transaction.gas_price for v in self.stats if v.included]
        return np.median(prices).tolist()


    def to_dict(self):
        total_blocks_waited = self.compute_total_blocks_waited()
        total_gas_price_diff = self.compute_total_gas_price_diff()
        total_count = len(self.stats)
        included_count = self.compute_included_count()
        return dict(
            total_count=total_count,
            included_count=included_count,
            average_gas_price=self.compute_average_gas_price(),
            median_gas_price=self.compute_median_gas_price(),
            not_included_count=total_count - included_count,
            total_gas_price_diff=total_gas_price_diff,
            total_blocks_waited=total_blocks_waited,
            average_blocks_waited=total_blocks_waited / included_count,
            average_gas_price_diff=total_gas_price_diff / included_count,
            results=[v.to_dict() for v in self.stats],
        )
