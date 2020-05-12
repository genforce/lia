"""Utility functions for reading and saving images."""

import glob
import numpy as np
import scipy
import scipy.misc
import cv2
from training.misc import adjust_dynamic_range


def preparing_data(im_path, img_type):
    """
    read images from the given path, and transform images from [0, 255] to [-1., 1.]

    return image shape: [N, C, H, W]
    """
    images = sorted(glob.glob(im_path + '/*' + img_type))
    images_name = []
    input_images = []
    for im_name in images:
        input_images.append(cv2.imread(im_name)[:, :, ::-1])
        images_name.append(im_name.split('/')[-1].split('.')[0])
    input_images = np.asarray(input_images)
    input_images = adjust_dynamic_range(input_images.astype(np.float32), [0, 255], [-1., 1.])
    input_images = input_images.transpose(0, 3, 1, 2)
    return input_images, images_name


def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    """
    transform images from [-1.0, 1.0] to [min_value, max_value] of dtype
    """
    assert \
        np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)


def imwrite(image, path):
    """ save an [-1.0, 1.0] image """

    if image.ndim == 3 and image.shape[2] == 1:  # for gray image
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]
    return scipy.misc.imsave(path, to_range(image, 0, 255, np.uint8))


def immerge(images, row, col):
    """
    merge images into an image with (row * h) * (col * w)

    `images` is in shape of N * H * W(* C=1 or 3)
    """
    h, w = images.shape[1], images.shape[2]
    if images.ndim == 4:
        img = np.zeros((h * row, w * col, images.shape[3]))
    elif images.ndim == 3:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    return img
