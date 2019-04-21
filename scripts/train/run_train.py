import argparse
import configparser

from train_setup import train


def preprocess_config(c):
    conf_dict = {}
    int_params = ['data.batch', 'model.z_dim', 'data.cuda', 'data.gpu',
                  'train.epochs']
    float_params = ['train.beta_1', 'train.beta_2',
                    'train.lr_discriminator', 'train.lr_generator']
    for param in c:
        if param in int_params:
            conf_dict[param] = int(c[param])
        elif param in float_params:
            conf_dict[param] = float(c[param])
        else:
            conf_dict[param] = c[param]
    return conf_dict


parser = argparse.ArgumentParser(description='Run training of DCGAN')
parser.add_argument("--config", type=str, default="./scripts/celeba.conf",
                    help="Path to the config file.")

# Run training
args = vars(parser.parse_args())
config = configparser.ConfigParser()
config.read(args['config'])
config = preprocess_config(config['TRAIN'])
train(config)
