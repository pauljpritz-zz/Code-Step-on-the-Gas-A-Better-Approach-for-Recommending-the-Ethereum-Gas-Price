from .pipeline.data_reader import read_data

def run_dummy(cnf: dict):
    read_data(cnf)