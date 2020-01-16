from .simple_gru import SimpleGRU
from .distribution_gru import DistributionGRU


def configure_model(cnf):
    cnf_model = cnf['model']

    if cnf['type'] == 'distribution':
        return DistributionGRU(hidden_size=cnf_model['hidden_size'],
                               input_size=cnf_model['input_size'],
                               num_layers=cnf_model["num_layers"],
                               pred_steps=cnf_model['pred_steps'],
                               linear_units=cnf_model['linear_units'],
                               dropout=cnf_model['dropout'])
    else:
        return SimpleGRU(hidden_size=cnf_model['hidden_size'],
                         input_size=cnf_model['input_size'],
                         num_layers=cnf_model["num_layers"],
                         pred_steps=cnf_model['pred_steps'],
                         linear_units=cnf_model['linear_units'],
                         dropout=cnf_model['dropout'])
