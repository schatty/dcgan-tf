import os
import unittest
from scripts import train


class TestFlowers(unittest.TestCase):

    def test_basic_flowers(self):
        config = {
            'data.datadir': 'data',
            'data.dataset': 'flowers',
            'data.batch': 32,
            'data.img_mode': 'RGB',
            'data.cuda': 1,
            'data.gpu': 0,
            'train.beta_1': 0.001,
            'train.beta_2': 0.999,
            'train.lr_discriminator': 0.001,
            'train.lr_generator': 0.001,
            'train.epochs': 1,
            'train.exp_dir': 'results/flowers',
            'train.log_dir': 'results/flowers/logs',
            'train.tb_dir': 'results/flowers/tensorboard/',

            'model.x_dim': '64,64,3',
            'model.z_dim': 500,
            'model.save_dir':'tests/results/models/flowers'
        }
        train(config)