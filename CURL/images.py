# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""Image loading and axis-order helpers."""
import numpy as np
from PIL import Image


def load_image(img_filepath: str, normaliser: float) -> np.ndarray:
    """Loads an image from file as a numpy multi-dimensional array

    :param img_filepath: filepath to the image
    :returns: image as a multi-dimensional numpy array
    :rtype: multi-dimensional numpy array

    """
    # Convert to RGB: the network takes three channels, and FiveK exports
    # include greyscale images and PNGs carrying an alpha channel.
    img = Image.open(img_filepath).convert('RGB')
    img = normalise_image(np.array(img), normaliser)  # NB: imread normalises to 0-1
    return img


def normalise_image(img: np.ndarray, normaliser: float) -> np.ndarray:
    """Normalises image data to be a float between 0 and 1

    :param img: Image as a numpy multi-dimensional image array
    :returns: Normalised image as a numpy multi-dimensional image array
    :rtype: Numpy array

    """
    img = img.astype('float32') / normaliser
    return img


def swapimdims_3HW_HW3(img: np.ndarray) -> np.ndarray:
    """Move the image channels to the first dimension of the numpy
    multi-dimensional array

    :param img: numpy nd array representing the image
    :returns: numpy nd array with permuted axes
    :rtype: numpy nd array

    """
    if img.ndim == 3:
        return np.swapaxes(np.swapaxes(img, 1, 2), 0, 2)
    elif img.ndim == 4:
        return np.swapaxes(np.swapaxes(img, 2, 3), 1, 3)


def swapimdims_HW3_3HW(img: np.ndarray) -> np.ndarray:
    """Move the image channels to the last dimensiion of the numpy
    multi-dimensional array

    :param img: numpy nd array representing the image
    :returns: numpy nd array with permuted axes
    :rtype: numpy nd array

    """
    if img.ndim == 3:
        return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
    elif img.ndim == 4:
        return np.swapaxes(np.swapaxes(img, 1, 3), 2, 3)
