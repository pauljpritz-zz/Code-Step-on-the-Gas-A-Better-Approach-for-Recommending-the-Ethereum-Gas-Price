import json
import gzip as gz
from datetime import datetime
import datetime as dt
from pprint import pprint
import pandas as pd


def read_data(cnf: dict):
    start_time = datetime.fromisoformat(cnf['data']['start_date']) - dt.timedelta(days=1.5)
    end_time = datetime.fromisoformat(cnf['data']['end_date'])

    eth_price_file = cnf['data']['eth_price_file']
    if eth_price_file.endswith('.csv'):
        eth_price = pd.read_csv(eth_price_file)
    elif eth_price_file.endswith('.json'):
        with open(eth_price_file) as f:
            eth_price = json.load(f)
    else:
        raise NotImplementedError

    gas_price = []
    with gz.open(cnf['data']['gas_price_file'], 'r') as f:
        if cnf['testing']:
            count = 0
            for line in f:
                gas_price.append(json.loads(line))
                count += 1
                if count >= 1000:
                    break
        else:
            for line in f:
                current = json.loads(line)
                if 'timestamp' in current:
                    time = datetime.fromtimestamp(current['timestamp'])
                    if time <= end_time:
                        gas_price.append(current)
                    if time < start_time:
                        break

    return eth_price, gas_price
