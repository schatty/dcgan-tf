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