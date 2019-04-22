import os
import unittest
from scripts import train


class TestMnist(unittest.TestCase):

    def test_basic_mnist(self):
        config = {
            'data.datadir': 'data',
            'data.dataset': 'mnist',
            'data.batch': 128,
            'data.img_mode': 'L',
            'data.cuda': 1,
            'data.gpu': 0,
            'train.beta_1': 0.001,
            'train.beta_2': 0.999,
            'train.lr_discriminator': 0.001,
            'train.lr_generator': 0.001,
            'train.epochs': 1,
            'train.exp_dir': 'results/mnist',
            'train.log_dir': 'results/mnist/logs',
            'train.tb_dir': 'results/mnist/tensorboard/',

            'model.x_dim': '28,28,1',
            'model.z_dim': 200,
            'model.save_dir':'tests/results/models/mnist'
        }
        train(config)