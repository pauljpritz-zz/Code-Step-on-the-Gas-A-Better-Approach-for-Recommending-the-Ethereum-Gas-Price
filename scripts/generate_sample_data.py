import gzip
import json
import datetime as dt

import numpy as np


def generate_sin_wave_noise(length: int = 5000, freq: int = 5, noise_var: float = 0.1):
    """
    Generates a synthetic time series consisting of a sin wave and gaussian noise.
    The series is normalised to lie between 0 and 1.
    :param length: Number of samples in the series.
    :param freq: Number of cycles in the entire series.
    :param noise_var: Variance of the Gaussian noise component.
    :return: Synthetic normalised time series.
    """
    x = np.arange(length)
    y = (np.sin(2 * np.pi * freq * x / length))
    y_noise = np.random.normal(0, noise_var, length)
    y = y + y_noise
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    return y


data = []
current_time = dt.datetime(2020, 1, 15)
end_time = dt.datetime(2020, 1, 16)
delta = dt.timedelta(seconds=14)
v = 0
while current_time < end_time:
    data.append(dict(timestamp=current_time.timestamp()))
    current_time += delta

sin_wave = generate_sin_wave_noise(len(data), freq=20, noise_var=0)
for i in range(len(data)):
    data[i]["average_gas_price"] = sin_wave[i]

with gzip.open("sample-data.jsonl.gz", "wt") as f:
    for datum in data:
        json.dump(datum, f)
        f.write("\n")
