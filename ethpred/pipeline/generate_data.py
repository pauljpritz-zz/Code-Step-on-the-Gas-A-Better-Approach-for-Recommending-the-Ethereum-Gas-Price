from .data_reader import read_data
from .to_pandas import convert_to_dataframe

def generate_data(cnf:dict):
     eth_price, gas_price = read_data(cnf)
     data = convert_to_dataframe(eth_price, gas_price, cnf['data'])
     print(data)