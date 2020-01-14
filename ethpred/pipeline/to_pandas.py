import pandas as pd


def convert_to_dataframe(eth_price: dict, gas_price: list):
    # TODO: Add to config
    features = ['average_gas_price', 'contracts_tx_count']

    gas_price_dict = {}
    for obs in range(len(gas_price)):
        gas_price_dict[obs] = {}
        for feature in features:
            if feature in gas_price[obs]:
                gas_price_dict[obs][feature] = gas_price[obs][feature]
            else:
                gas_price_dict[obs][feature] = None

    gas_price_df = pd.DataFrame.from_dict(gas_price_dict, orient='index')

    # TODO: How will we match the two datasets?
    return gas_price_df
