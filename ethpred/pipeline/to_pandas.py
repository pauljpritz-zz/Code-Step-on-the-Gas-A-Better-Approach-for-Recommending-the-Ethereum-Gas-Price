from datetime import datetime
import pandas as pd
from .calc_distributions import generate_distribution


def convert_to_dataframe(eth_prices: dict, gas_price: list, cnf: dict):
    features = cnf['data']['features']
    nested_features = cnf['data']['nested_features']

    gas_price_dict = {}
    for obs in range(len(gas_price)):
        gas_price_dict[obs] = {}
        for feature in features:
            if feature in gas_price[obs]:
                gas_price_dict[obs][feature] = gas_price[obs][feature]
            else:
                gas_price_dict[obs][feature] = None
        if 'timestamp' in gas_price[obs]:
            gas_price_dict[obs]['time'] = datetime.fromtimestamp(gas_price[obs]['timestamp'])

        for feature in nested_features:
            key = list(feature.keys())[0]
            val = feature[key]
            if key in gas_price[obs] and val in gas_price[obs][key]:
                gas_price_dict[obs][key] = gas_price[obs][key][val]
            else:
                gas_price_dict[obs][key] = None

        if cnf['type'] == 'distribution' and 'transactions' in gas_price[obs]:
            for i in range(len(gas_price[obs]['transactions'])):
                gas_price_dict[obs]['mean'], gas_price_dict[obs]['std_dev'] = generate_distribution(
                    gas_price[obs]['transactions'])
                gas_price_dict[obs]['gas_price_' + str(i)] = gas_price[obs]['transactions'][i][
                    'gas_price']

    # print(gas_price_dict)
    eth_price_df = pd.DataFrame.from_dict(eth_prices)
    eth_price_df['date'] = pd.to_datetime(eth_price_df['date'], format='%Y-%m-%d')
    eth_price_df = eth_price_df[cnf['data']['eth_price_features']]

    gas_price_df = pd.DataFrame.from_dict(gas_price_dict, orient='index')
    # print(gas_price_df.head())
    gas_price_df = gas_price_df.dropna(axis='rows', subset=['time']).fillna(0).sort_values(
        by='time')
    eth_price_df = eth_price_df.dropna(axis='rows', subset=['date']).fillna(0).sort_values(
        by='date')
    # print("Gas price:", gas_price_df)
    # print("ETH price:", eth_price_df)

    data = pd.merge_asof(gas_price_df, eth_price_df, left_on='time', right_on='date',
                         direction='backward')
    # print("data:", data)

    data = data[
        (data['time'] > cnf['data']['start_date']) & (data['time'] <= cnf['data']['end_date'])]
    data = data.drop(columns=['time', 'date'])

    print(data.head())

    return data
