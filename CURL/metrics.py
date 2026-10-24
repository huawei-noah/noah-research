# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""Image quality metrics, over numpy arrays in [0, 1]."""
import inspect

import numpy as np
from skimage.metrics import structural_similarity as ssim

from images import swapimdims_3HW_HW3

# scikit-image renamed `multichannel=True` to `channel_axis=-1` in 0.19 and
# removed the old name in 0.23. Ask the installed function which it takes.
_SSIM_COLOUR_KWARG = (
    {'channel_axis': -1}
    if 'channel_axis' in inspect.signature(ssim).parameters
    else {'multichannel': True}
)


def compute_mse(original: np.ndarray, result: np.ndarray) -> float:
    """Computes the mean squared error between to RGB images represented as multi-dimensional numpy arrays.

    :param original: input RGB image as a numpy array
    :param result: target RGB image as a numpy array
    :returns: the mean squared error between the input and target images
    :rtype: float

    """
    return ((original - result) ** 2).mean()


def compute_psnr(image_batchA: np.ndarray, image_batchB: np.ndarray,
                 max_intensity: float) -> float:
    """Computes the PSNR for a batch of input and output images

    :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
    :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
    :param max_intensity: maximum intensity possible in the image (e.g. 255)
    :returns: average PSNR for the batch of images
    :rtype: float

    """
    num_images = image_batchA.shape[0]
    psnr_val = 0.0

    for i in range(0, num_images):
        imageA = image_batchA[i, 0:3, :, :]
        imageB = image_batchB[i, 0:3, :, :]
        imageB = np.maximum(0, np.minimum(imageB, max_intensity))
        mse = compute_mse(imageA, imageB)
        # An exact reconstruction gives mse == 0 and an infinite PSNR,
        # which poisons the running average and leaves best_valid_psnr
        # unbeatable for the rest of the run. Floor it instead.
        mse = max(float(mse), 1e-10)
        psnr_val += 10 * np.log10(max_intensity ** 2 / mse)

    return psnr_val / num_images


def compute_ssim(image_batchA: np.ndarray, image_batchB: np.ndarray) -> float:
    """Computes the SSIM for a batch of input and output images

    :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
    :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
    :param max_intensity: maximum intensity possible in the image (e.g. 255)
    :returns: average PSNR for the batch of images
    :rtype: float

    """
    num_images = image_batchA.shape[0]
    ssim_val = 0.0

    for i in range(0, num_images):
        imageA = swapimdims_3HW_HW3(
            image_batchA[i, 0:3, :, :])
        imageB = swapimdims_3HW_HW3(
            image_batchB[i, 0:3, :, :])
        # data_range is the range of the *representation*, not of this
        # particular image: images are normalised to [0, 1]. Passing
        # imageA.max()-imageA.min() made a low-contrast target score
        # differently from a high-contrast one, so reported SSIM was not
        # comparable between images or with published figures.
        ssim_val += ssim(imageA, imageB, data_range=1.0,
                         gaussian_weights=True, win_size=11, **_SSIM_COLOUR_KWARG)

    return ssim_val / num_images
