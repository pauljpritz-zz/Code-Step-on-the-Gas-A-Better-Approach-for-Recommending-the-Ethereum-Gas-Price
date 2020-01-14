from .pipeline.generate_data import generate_data

def run_dummy(cnf: dict):
    generate_data(cnf)