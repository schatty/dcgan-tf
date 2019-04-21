import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Bidirectional, \
    BatchNormalization, LeakyReLU, MaxPool2D, Conv2DTranspose, Reshape
from tensorflow.keras import Model
from tensorflow.keras.activations import tanh


class DCGAN(Model):
    def __init__(self):
        super(DCGAN, self).__init__()

        out_channel_dim = 1
        # Leaky relu parameter
        self.alpha = 0.2

        # Discriminator
        self.d = tf.keras.Sequential([
            Conv2D(64, 3, strides=2, padding='same'),
            LeakyReLU(self.alpha),

            Conv2D(64, 3, strides=2, padding='same'),
            LeakyReLU(self.alpha),

            Conv2D(128, 3, strides=2, padding='same'),
            BatchNormalization(),
            LeakyReLU(self.alpha),

            Conv2D(256, 3, strides=1, padding='same'),
            BatchNormalization(),
            LeakyReLU(self.alpha),

            Dense(1, kernel_initializer=tf.initializers.GlorotNormal())
        ])

        # Generator
        self.g = tf.keras.Sequential([
            Dense(7*7*256),
            Reshape((7, 7, 256)),
            LeakyReLU(self.alpha),

            Conv2DTranspose(128, 5, strides=2, padding='same'),
            BatchNormalization(),
            LeakyReLU(self.alpha),

            Conv2DTranspose(64, 5, strides=1, padding='same'),
            BatchNormalization(),
            LeakyReLU(self.alpha),

            Conv2DTranspose(1, 5, strides=2, padding='same'),
        ])

    def call(self, input_real, input_z):
        input_real = tf.cast(input_real, tf.float32)
        input_z = tf.cast(input_z, tf.float32)

        # Generate new image
        g_model = tanh(self.g(input_z))
        # Discriminate real images as real
        d_logits_real = self.d(input_real)
        # Discriminate fake images
        d_logits_fake = self.d(g_model)

        smooth = 0.01
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                    labels=tf.ones_like(d_logits_real) * (1 - smooth))
        )
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                    labels=tf.zeros_like(d_logits_fake))
        )
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                    labels=tf.ones_like(d_logits_fake) * (1 - smooth))
        )

        # Final discriminator loss is the sum of real and fake discriminators
        d_loss = d_loss_real + d_loss_fake

        return d_loss, g_loss


    def get_generator_output(self, n_images, z_dim, image_mode):
        c_map = None if image_mode == 'RGB' else 'gray'
        example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

        samples = tf.keras.activations.tanh(self.g(example_z))
        return samples