import os
import shutil
from urllib.request import urlretrieve
import numpy as np
from PIL import Image
import gzip
from tqdm import tqdm


def _read32(bytestream):
    """
    Read 32-bit integer from bytesteam
    Args:
        bytestream (bytes): A bytestream
    Return (int): 32-bit integer
    """
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def _ungzip(save_path, extract_path, database_name, *args):
    """
    Extract gzip file into extract_path
    Args:
        save_path (str): path to the gzip file
        extract_path (str): path to extract data to
        database_name (str): database name

    Return: None
    """
    # Get data from save_path
    with open(save_path, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            if magic != 2051:
                raise ValueError(
                    'Invalid magic number {} in file: {}'.format(magic,
                                                                 f.name))
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols)

    # Save data to extract_path
    for image_i, image in enumerate(
            tqdm(data, unit='File', unit_scale=True, miniters=1,
                 desc='Extracting {}'.format(database_name))):
        Image.fromarray(image, 'L').save(
            os.path.join(extract_path, 'image_{}.jpg'.format(image_i)))


def download_dataset(db_name='mnist', dist_path='data'):
    """Download image database from given name"""
    db_name = db_name.lower()
    assert db_name in ['mnist'], "Unknown database name"

    if db_name == 'mnist':
        url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        extract_path = os.path.join(dist_path, 'mnist')
        save_path = os.path.join(dist_path, 'train-images-idx3-ubyte.gz')
        extract_fn = _ungzip

    if os.path.exists(extract_path):
        print('Found {} Data'.format(db_name))
        return

    if not os.path.exists(dist_path):
        os.makedirs(dist_path)

    if not os.path.exists(save_path):
        print("Downloading...")
        urlretrieve(url, save_path)

    os.makedirs(extract_path)
    try:
        extract_fn(save_path, extract_path, db_name, dist_path)
    except Exception as err:
        shutil.rmtree(
            extract_path)  # Remove extraction folder if there is an error
        raise err

    # Remove compressed data
    os.remove(save_path)


def get_batch(image_files, width, height, mode):
    """
    Get batch of images.

    Args:
        image_files (list): list of paths to the images
        width (int): image width
        height (int): image height
        mode (str): Pillow image mode

    Returns:

    """
    data_batch = np.array(
        [get_image(sample_file, width, height, mode) for sample_file in image_files]
    ).astype(np.float32)

    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape((-1, width, height, 1))
    return data_batch


def get_image(image_path, width, height, mode):
    """
    Read image from image path.

    Args:
        image_path (str): path to the image
        width (int): image path
        height (int): image height
        mode (str): Pillow image mode

    Returns (np.ndarray): numpy ndarray

    """
    image = Image.open(image_path)
    return np.asarray(image.convert(mode))


def preprocess_mnist(data):
    return data / 255.


def image_grid(images, rows, cols, w, h, c, mode):
    """
    Get Pillow image object with grid of images.

    Args:
        images (np.ndarray): numpy array
        rows (int): number of rows in a grid
        cols (int): number of cols in a grid
        w (int): image width
        h (int): image height
        c (c): number of channels
        mode (str): Pillow mode

    Returns (PIL.Image): image object

    """
    assert len(images) == rows * cols, "Number of images should be multiple of rows*cols"

    # Scale to 0-255
    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

    # Put images in a square arrangement
    w_orig = images.shape[1]
    h_orig = images.shape[2]
    images = images.reshape((rows, cols, w_orig, h_orig, c))

    # Combine images to grid image
    new_im = Image.new(mode, (w*cols, h*rows))
    for i_row in range(rows):
        for i_col in range(cols):
            image = images[i_row, i_col, :, :, :].squeeze()
            im = Image.fromarray(image, mode).resize((w, h))
            new_im.paste(im, (i_col * w, i_row * h))

    return new_im


class Dataset(object):
    """Dataset loader object"""

    def __init__(self, db_name, data_files):
        IMAGE_WIDTH = 28
        IMAGE_HEIGHT = 28

        db_name = db_name.lower()
        if db_name == "mnist":
            self.image_mode = 'L'
            image_channels = 1

        self.data_files = data_files
        self.shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, image_channels

    def get_batches(self, batch_size):
        """Generate batches of data"""

        current_index = 0
        while current_index + batch_size <= self.shape[0]:
            data_batch = get_batch(
                self.data_files[current_index:current_index+batch_size],
                *self.shape[1:3],
                self.image_mode
            )
            current_index += batch_size
            yield data_batch / 255. - 0.5