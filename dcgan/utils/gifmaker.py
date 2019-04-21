import os
from glob import glob
from PIL import Image


def make_gif(source_dir, output):
    """
    Make gif file from set of .jpeg images.
    Args:
        source_dir (str): path with .jpeg images
        output (str): path to the output .gif file

    Returns: None

    """
    batch_sort = lambda s: int(s[s.index('-')+1:s.index('.')])
    image_paths = sorted(glob(os.path.join(source_dir, "*.jpg")),
                         key=batch_sort)
    frames = []
    for path in image_paths:
        img = Image.open(path)
        frames.append(img)
    frames[0].save(output, format='GIF', append_images=frames[1:],
                   save_all=True, duration=5*len(frames), loop=0)


if __name__ == "__main__":
    make_gif(source_dir="results/celeba/gen_output",
             output="results/celeba/gen_output/gen.gif")