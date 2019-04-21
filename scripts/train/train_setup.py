import os
import numpy as np
from glob import glob
import datetime
import time
from shutil import copyfile

import logging
real_log = f"{datetime.datetime.now():%Y-%m-%d_%H:%M}.log"
logging.basicConfig(filename=real_log,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y.%m.%d %H:%M:%S',
                    level=logging.DEBUG)
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

import tensorflow as tf
tf.config.gpu.set_per_process_memory_growth(True)

from dcgan.data import download_dataset, get_batch, preprocess_mnist, image_grid, Dataset
from dcgan.model import DCGAN
from dcgan import TrainEngine


def scale_img(x):
    # Rescale from [-0.5, 0.5] to [-1, 1]
    return x * 2


def train(config):
    np.random.seed(2019)
    tf.random.set_seed(2019)

    # Create folder for model
    model_dir = config['model.save_dir'][:config['model.save_dir'].rfind('/')]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create model for logs
    log_dir = config['train.log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_fn = f"{config['data.dataset']}_{datetime.datetime.now():%Y-%m-%d_%H:%M}.log"
    log_fn = os.path.join(log_dir, log_fn)
    gen_output_path = os.path.join(config['train.exp_dir'], 'gen_output')
    if not os.path.exists(gen_output_path):
        os.makedirs(gen_output_path)
    print(f"All info about training can be found in {log_fn}")

    # Determine device
    if tf.test.is_gpu_available():
        cuda_num = 0
        device_name = f'GPU:{cuda_num}'
    else:
        print("Training on CPU")
        device_name = 'CPU:0'

    # TODO: move to the Dataset constructor
    download_dataset("MNIST")

    img_w, img_h, img_c = list(map(int, config['model.x_dim'].split(',')))
    print("Image width: ", img_w)
    print("Image height: ", img_h)
    print("Image channels: ", img_c)

    beta_1 = config['train.beta_1']
    beta_2 = config['train.beta_2']
    lr_g = config['train.lr_generator']
    g_optimizer = tf.optimizers.Adam(lr_g, beta_1=beta_1, beta_2=beta_2)
    lr_d = config['train.lr_discriminator']
    d_optimizer = tf.optimizers.Adam(lr_d, beta_1=beta_1, beta_2=beta_2)

    # Metrics to gather
    train_g_loss = tf.metrics.Mean(name='g_loss')
    train_d_loss = tf.metrics.Mean(name='d_loss')

    # Summary writers
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    train_log_dir = config['train.tb_dir'] + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    model = DCGAN()

    def train_step(input_real, input_z, step):
        # Train step for generator
        with tf.GradientTape(persistent=True) as tape:
            d_loss, g_loss = model(input_real, input_z)

        # Discriminator update
        d_gradients = tape.gradient(d_loss, model.d.trainable_variables)
        d_optimizer.apply_gradients(
            zip(d_gradients, model.d.trainable_variables)
        )

        # Generator update
        g_gradients = tape.gradient(g_loss, model.g.trainable_variables)
        g_optimizer.apply_gradients(
            zip(g_gradients, model.g.trainable_variables)
        )
        del tape

        train_g_loss(g_loss)
        train_d_loss(d_loss)

    # Main training loop
    batch_size = config['data.batch']
    z_dim = config['model.z_dim']
    epochs = config['train.epochs']
    dataset_name = config['data.dataset']
    data_dir = config['data.datadir']
    mnist_dataset = Dataset(dataset_name, glob(os.path.join(data_dir, "mnist/*.jpg")))
    n_samples, img_width, img_height, n_channels = mnist_dataset.shape

    print("n_samples: ", n_samples)
    print("img_width: ", img_width)
    print("img_height: ", img_height)
    print("n_channels: ", n_channels)

    train_engine = TrainEngine()

    # Set hooks on training engine

    def on_start(state):
        logging.info("Training started.")
        train_g_loss.reset_states()
        train_d_loss.reset_states()
    train_engine.hooks['on_start'] = on_start

    def on_end(state):
        logging.info("Training ended.")
    train_engine.hooks['on_end'] = on_end

    def on_start_epoch(state):
        logging.info(f"Epoch {state['epoch']} started.")
    train_engine.hooks['on_start_epoch'] = on_start_epoch

    def on_end_epoch(state):
        pass
    train_engine.hooks['on_end_epoch'] = on_end_epoch

    def on_start_batch(state):
        step = state['step']

        batch_images = state['sample']
        # Rescale batch images
        batch_images = scale_img(batch_images)

        # Sample random noise from generator
        batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

        # Make training step
        train_step(batch_images, batch_z, step=state['step'])

        # Log losses on batch
        with train_summary_writer.as_default():
            tf.summary.scalar('g_loss', train_g_loss.result(), step=step)
            tf.summary.scalar('d_loss', train_d_loss.result(), step=step)

        if state['step'] % 100 == 0:
            # TODO: Remove
            print("Epoch {}/{}. Batch {}".format(state['epoch'], epochs,
                                                 state['step']))
            gen_output = model.get_generator_output(8, 200, 'L').numpy()
            images_grid = image_grid(gen_output, 2, 4, 28, 28, 1, 'L')
            images_grid.save(f"{gen_output_path}/{state['epoch']}-{state['step']}.jpg")
    train_engine.hooks['on_start_batch'] = on_start_batch

    def on_end_batch(state):
        step = state['step']
        msg = "Epoch {:4d} Batch {:6d} Generator {:10.6f} Discriminator {:10.6f}"
        logging.info(msg.format(state['epoch'], step, train_g_loss.result(),
                                train_d_loss.result()))
        train_g_loss.reset_states()
        train_d_loss.reset_states()
    train_engine.hooks['on_end_batch'] = on_end_batch

    time_start = time.time()

    with tf.device(device_name):
        train_engine.train(
            loader=mnist_dataset,
            epochs=epochs,
            batch=batch_size
        )

    time_end = time.time()
    elapsed = time_end - time_start
    h, min = elapsed // 3600, elapsed % 3600 // 60
    sec = elapsed - min * 60
    logging.info(f"Training took: {h} h {min} min {sec} sec")
    copyfile(real_log, log_fn)
    os.remove(real_log)