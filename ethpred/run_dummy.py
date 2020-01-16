from .pipeline.generate_data import generate_data
from .models.configure_model import configure_model
from .training.training_loops import GRU_training


def run_dummy(cnf: dict):
    train, test = generate_data(cnf)
    model = configure_model(cnf)
    GRU_training(model, train, test, cnf['training'])
