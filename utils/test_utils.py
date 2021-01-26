import argparse
from omegaconf import OmegaConf
from unittest import TestCase


class Test(TestCase):
    def setUp(self):
        # Arguments
        parser = argparse.ArgumentParser(description='Description...')
        # Load the config file
        parser.add_argument('--config', type=str, default='../configs/config.yml')
        flags = parser.parse_args()
        params = OmegaConf.load(flags.config)

    def test_train_test_split_data_dict(self):
        a = 7
        self.assertEqual(self.params['input_shape'],(28,28))
