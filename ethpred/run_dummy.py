from .pipeline.generate_data import generate_data
from .models.configure_model import configure_model
from .training.training_loops import GRU_training
from .training.logger import Logger


def run_dummy(cnf: dict):
    train, test = generate_data(cnf)
    model = configure_model(cnf)

    logger = Logger(cnf)

    GRU_training(model=model,
                 train_dataloader=train,
                 test_dataloader=test,
                 cnf=cnf['training'],
                 logger=logger)
