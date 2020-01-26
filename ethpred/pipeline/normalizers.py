import numpy as np


class MinMaxNormalizer:
    def __init__(self):
        self._min_value = None
        self._max_value = None

    @property
    def min_value(self):
        if self._min_value is None:
            raise ValueError("you must fit first")
        return self._min_value

    @property
    def max_value(self):
        if self._max_value is None:
            raise ValueError("you must fit first")
        return self._max_value

    def fit(self, array):
        self._min_value = np.min(array)
        self._max_value = np.max(array)

    def transform(self, array):
        result = (array - self.min_value) / (self.max_value - self.min_value)
        return result.fillna(0)

    def fit_transform(self, array):
        self.fit(array)
        return self.transform(array)

    def inverse_transform(self, array):
        return array * (self.max_value - self.min_value) + self.min_value


class LogNormalizer:
    def fit(self, array):
        # nothing to do
        pass

    def fit_transform(self, array):
        return self.transform(array)

    def transform(self, array):
        col = np.log(array)
        col[np.isneginf(col)] = 0
        return col

    def inverse_transform(self, array):
        return np.exp(array)


def create(name):
    return dict(
        log=LogNormalizer,
        minmax=MinMaxNormalizer,
    )[name]()
