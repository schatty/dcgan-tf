import numpy as np
from scipy.ndimage import rotate


def translate(img, shift=10, direction='right', roll=True):
    img = img.copy()
    if direction == 'right':
        right_slice = img[:, -shift:].copy()
        img[:, shift:] = img[:, :-shift]
        if roll:
            img[:,:shift] = np.fliplr(right_slice)
    if direction == 'left':
        left_slice = img[:, :shift].copy()
        img[:, :-shift] = img[:, shift:]
        if roll:
            img[:, -shift:] = left_slice
    if direction == 'down':
        down_slice = img[-shift:, :].copy()
        img[shift:, :] = img[:-shift,:]
        if roll:
            img[:shift, :] = down_slice
    if direction == 'up':
        upper_slice = img[:shift, :].copy()
        img[:-shift, :] = img[shift:, :]
        if roll:
            img[-shift:,:] = upper_slice
    return img


def gaussian_noise(img, mean=0, sigma=10):
    img = img.copy().astype(np.int16)
    noise = np.random.normal(mean, sigma, img.shape).astype(np.int16)
    mask_overflow_upper = img+noise >= 255
    mask_overflow_lower = img+noise < 0
    noise[mask_overflow_upper] = 0
    noise[mask_overflow_lower] = 0
    img += noise
    return img


def rotate_img(img, angle, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img


def get_augmented_images(img, shift=15, sigma=10):
    angles = [-30, -10, 0, 10, 30]
    aug_imgs = []
    for angle in angles:
        for trans_dir in ['left', 'right', 'up', 'down']:
            aug_imgs.append(gaussian_noise(translate(rotate_img(img, angle), trans_dir, shift), mean=0, sigma=sigma))
    return aug_imgs