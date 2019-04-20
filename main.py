import numpy as np
import os
from glob import glob
import tensorflow as tf
tf.config.gpu.set_per_process_memory_growth(True)

from data import download_dataset, get_batch, preprocess_mnist


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
