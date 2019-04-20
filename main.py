import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import tensorflow as tf

tf.config.gpu.set_per_process_memory_growth(True)

from data import download_dataset, get_batch, preprocess_mnist, image_grid, Dataset
from model import DCGAN


def scale_img(x):
    # Rescale from [-0.5, 0.5] to [-1, 1]
    return x * 2


if __name__ == "__main__":
    # Determine device
    if tf.test.is_gpu_available():
        print("GPU available")
        cuda_num = 0
        device_name = f'GPU:{cuda_num}'
    else:
        print("Training on CPU")
        device_name = 'CPU:0'

    download_dataset("MNIST")

    data_dir = 'data'
    image_paths = glob(os.path.join(data_dir, 'mnist/*.jpg'))[:10]
    image_batch = get_batch(image_paths, width=28, height=28, mode="L")
    image_batch = preprocess_mnist(image_batch)
    print(image_batch.shape, np.max(image_batch))

    lr_g = 0.001
    g_optimizer = tf.keras.optimizers.Adam(lr_g)
    lr_d = 0.001
    d_optimizer = tf.keras.optimizers.Adam(lr_d)

    # Metrics to gather
    train_g_loss = tf.metrics.Mean(name='g_loss')
    train_d_loss = tf.metrics.Mean(name='d_loss')

    model = DCGAN()

    def train_step(input_real, input_z):
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

        # Log loss for step
        train_g_loss(g_loss)
        train_d_loss(d_loss)

    # Main training loop
    batch_size = 128
    z_dim = 200
    learning_rate = 0.001
    beta1 = 0.001

    epochs = 5
    mnist_dataset = Dataset("mnist", glob(os.path.join(data_dir, "mnist/*.jpg")))
    n_samples, img_width, img_height, n_channels = mnist_dataset.shape

    print("n_samples: ", n_samples)
    print("img_width: ", img_width)
    print("img_height: ", img_height)
    print("n_channels: ", n_channels)

    with tf.device(device_name):
        n_batches = 0
        losses = []
        for i_epoch in range(epochs):
            train_d_loss.reset_states()
            train_g_loss.reset_states()

            for batch_images in mnist_dataset.get_batches(batch_size):
                n_batches += 1

                # Rescale batch images
                batch_images = scale_img(batch_images)

                # Sample random noise from generator
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

                # Make training step
                train_step(batch_images, batch_z)

                if n_batches % 100 == 0:
                    print("Epoch {}/{}. Batch {}".format(i_epoch + 1, epochs,
                                                         n_batches))
                    gen_output = model.get_generator_output(8, 200, 'L').numpy()
                    images_grid = image_grid(gen_output, 2, 4, 28, 28, 1, 'L')
                    images_grid.save(f"results/gen_output/{i_epoch}-{n_batches}.jpg")

            print("Epoch ", i_epoch,
                  "Generator loss: ", train_g_loss.result(),
                  "Discriminator loss: ", train_d_loss.result())