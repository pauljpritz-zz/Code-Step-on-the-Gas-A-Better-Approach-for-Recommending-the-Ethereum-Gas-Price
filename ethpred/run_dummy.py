from .pipeline.generate_data import generate_data
from .models.simple_gru import SimpleGRU, configure_SimpleGRU
from .training.training_loops import GRU_training

def run_dummy(cnf: dict):
    train, test = generate_data(cnf)
    model = configure_SimpleGRU(cnf['model'])
    GRU_training(model, train, test, cnf['training'])
