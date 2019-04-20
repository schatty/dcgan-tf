import numpy as np
import tensorflow as tf
tf.config.gpu.set_per_process_memory_growth(True)

from data import download_dataset


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

