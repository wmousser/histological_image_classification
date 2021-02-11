import argparse
from omegaconf import OmegaConf
from utils.utils import *
from utils.utils_breakHis import *

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Description...')
    # Load the config file
    parser.add_argument('--config', type=str, default='./configs/config.yml')
    flags = parser.parse_args()
    params = OmegaConf.load(flags.config)
    print_dict_values('params', params)

    # prepare data
    x_train, y_train, x_test, y_test, x_valid, y_valid = prepare_data(params)

    # prepare the model and fit
    model = prepare_model(x_train, y_train, x_valid, y_valid, params)

    # predictions
    perform_predictions(model, x_test, y_test, params)
    print('Confusion matrix saved at %s'%params['log_dir']+ params['model_name'])
