import numpy as np


def generate_distribution(transactions):
    gas_prices = np.array([i['gas_price'] for i in transactions])
    return np.mean(gas_prices), np.std(gas_prices)
