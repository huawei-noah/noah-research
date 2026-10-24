# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""Compatibility shim for the old ``util.ImageProcessing`` namespace.

The colour conversions, curve arithmetic, metrics and image helpers now live in
:mod:`colour`, :mod:`curves`, :mod:`metrics` and :mod:`images`. Import them from
there. This module keeps ``ImageProcessing`` working for code written against
the previous layout.
"""
from colour import hsv_to_rgb, lab_to_rgb, rgb_to_hsv, rgb_to_lab
from curves import adjust_hsv, adjust_lab, adjust_rgb, apply_curve
from images import (load_image, normalise_image, swapimdims_3HW_HW3,
                    swapimdims_HW3_3HW)
from metrics import compute_mse, compute_psnr, compute_ssim

__all__ = ['ImageProcessing', 'rgb_to_lab', 'lab_to_rgb', 'rgb_to_hsv',
           'hsv_to_rgb', 'apply_curve', 'adjust_lab', 'adjust_rgb',
           'adjust_hsv', 'compute_mse', 'compute_psnr', 'compute_ssim',
           'load_image', 'normalise_image', 'swapimdims_3HW_HW3',
           'swapimdims_HW3_3HW']


class ImageProcessing(object):
    """The previous namespace, delegating to the modules above."""

    rgb_to_lab = staticmethod(rgb_to_lab)
    lab_to_rgb = staticmethod(lab_to_rgb)
    rgb_to_hsv = staticmethod(rgb_to_hsv)
    hsv_to_rgb = staticmethod(hsv_to_rgb)
    apply_curve = staticmethod(apply_curve)
    adjust_lab = staticmethod(adjust_lab)
    adjust_rgb = staticmethod(adjust_rgb)
    adjust_hsv = staticmethod(adjust_hsv)
    compute_mse = staticmethod(compute_mse)
    compute_psnr = staticmethod(compute_psnr)
    compute_ssim = staticmethod(compute_ssim)
    load_image = staticmethod(load_image)
    normalise_image = staticmethod(normalise_image)
    swapimdims_3HW_HW3 = staticmethod(swapimdims_3HW_HW3)
    swapimdims_HW3_3HW = staticmethod(swapimdims_HW3_3HW)
