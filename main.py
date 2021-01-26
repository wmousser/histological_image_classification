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

    # data = prepare_data(params)
    prepare_data(params)

    # model = prepare_model(params)

    # fit(model, data)

    # display_results(history)