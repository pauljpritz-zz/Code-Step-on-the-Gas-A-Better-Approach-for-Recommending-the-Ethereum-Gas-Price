import json
import numpy as np
import pandas as pd
from pprint import pprint


def read_data(cnf: dict):
    with open(cnf['data']['eth_price_file']) as f:
        data = json.load(f)
    pprint(data)


    with open(cnf['data']['gas_price_file']) as f:
        for line in f:
            data = json.loads(line)
            pprint(data)