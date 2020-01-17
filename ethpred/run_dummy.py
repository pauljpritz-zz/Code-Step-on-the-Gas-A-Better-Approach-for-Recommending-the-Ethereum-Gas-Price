from .pipeline.generate_data import generate_data, create_dataloaders
from .models.configure_model import configure_model
from .training.training_loops import GRU_training
from .training.logger import Logger


def run_dummy(cnf: dict):
    X_train, X_test, y_train, y_test = generate_data(cnf)
    train, test = create_dataloaders(X_train, y_train, X_test, y_test, cnf)
    model = configure_model(cnf)

    logger = Logger(cnf)

    GRU_training(model=model,
                 train_dataloader=train,
                 test_dataloader=test,
                 cnf=cnf['training'],
                 logger=logger)
