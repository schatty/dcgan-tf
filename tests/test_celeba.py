import os
import unittest
from scripts import train


class TestCeleba(unittest.TestCase):

    def test_basic_celeba(self):
        config = {
            'data.datadir': 'data',
            'data.dataset': 'celeba',
            'data.batch': 32,
            'data.img_mode': 'RGB',
            'data.cuda': 1,
            'data.gpu': 0,
            'train.beta_1': 0.001,
            'train.beta_2': 0.999,
            'train.lr_discriminator': 0.001,
            'train.lr_generator': 0.001,
            'train.epochs': 1,
            'train.exp_dir': 'results/celeba',
            'train.log_dir': 'results/celeba/logs',
            'train.tb_dir': 'results/celeba/tensorboard/',
            'train.test_mode': True,
            'model.x_dim': '28,28,3',
            'model.z_dim': 500,
            'model.save_dir':'tests/results/models/celeba'
        }
        train(config)
