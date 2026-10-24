# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""The neural curve layer's arithmetic.

A curve is a piecewise linear function through sixteen equally spaced knots.
The heads predict the knot values; these functions evaluate the curve from them
and apply it to a channel."""
from typing import Tuple

import torch

from device import DEVICE
from tensors import channels_last, monotone_mapping

#: When True the curve heads are read as monotone tone mappings rather than
#: scaling curves, so the realised mapping cannot invert. Set by --monotone.
MONOTONE = False


def set_monotone(enabled: bool) -> None:
    """Selects the curve parameterisation for the whole run."""
    global MONOTONE
    MONOTONE = enabled


def _raw_split(params, count):
    """Splits a flat head output into `count` pieces, without exponentiating."""
    width = params.shape[-1] // count
    return [params[..., i * width:(i + 1) * width] for i in range(count)]


def _apply(img, params, slope_sqr_diff, channel_in, channel_out):
    """Dispatches to the mapping or the scaling parameterisation."""
    if MONOTONE:
        return apply_mapping(img, monotone_mapping(params), slope_sqr_diff,
                             channel_in, channel_out)
    return apply_curve(img, torch.exp(params), slope_sqr_diff,
                       channel_in, channel_out)


def apply_mapping(img: torch.Tensor, M: torch.Tensor, slope_sqr_diff: torch.Tensor,
                  channel_in: int, channel_out: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies a monotone tone mapping, read off at the knots and interpolated.

    M holds the mapping's value at each knot, built by
    :func:`tensors.monotone_mapping`, so it is increasing and spans [0, 1]. The
    output channel takes the interpolated value directly rather than being
    multiplied by a scaling factor, which is what keeps the result monotone.
    """
    steps = M.shape[-1] - 1
    batched = M.dim() == 2

    slope = M[..., 1:] - M[..., :-1]
    slope_sqr_diff = slope_sqr_diff + (
        (slope[..., 1:] - slope[..., :-1]) ** 2).sum()

    idx = torch.arange(slope.shape[-1], device=img.device, dtype=img.dtype)
    hinge = torch.clamp(
        img[..., channel_in].unsqueeze(-1) * steps - idx, 0, 1)
    if batched:
        extra = (1,) * (hinge.dim() - 2)
        mapped = (M[:, 0].view(-1, *extra)
                  + (hinge * slope.view(-1, *extra, slope.shape[-1])).sum(-1))
    else:
        mapped = M[0] + (hinge * slope).sum(-1)

    img_copy = img.clone()
    img_copy[..., channel_out] = mapped
    img_copy = torch.clamp(img_copy, 0, 1)
    return img_copy, slope_sqr_diff


def apply_curve(img: torch.Tensor, C: torch.Tensor, slope_sqr_diff: torch.Tensor,
                channel_in: int, channel_out: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies a piecewise linear curve defined by a set of knot points to
    an image channel

    The knots are equally spaced across [0, 1] in the driving channel, and
    C holds their values, so the curve passes through C[k] at x = k/steps.

    :param img: image to be adjusted, HxWx3
    :param C: predicted knot points of the curve
    :param slope_sqr_diff: running smoothness penalty, added to and returned
    :param channel_in: channel whose intensity drives the curve
    :param channel_out: channel the resulting scale is applied to
    :returns: adjusted image, updated smoothness penalty
    :rtype: Tensor, Tensor

    """
    # C is either one knot vector (K,) for a single image, or one per image
    # in the batch (B, K). Everything below broadcasts over the leading
    # dimension, so a batch gets its own curve per image rather than image
    # zero's curve applied to all of them.
    batched = C.dim() == 2
    curve_steps = C.shape[-1]-1

    '''
    Compute the slope of the line segments
    '''
    slope = C[..., 1:] - C[..., :-1]

    '''
    Compute the squared difference between slopes
    '''
    slope_sqr_diff = slope_sqr_diff + (
        (slope[..., 1:] - slope[..., :-1]) ** 2).sum()

    '''
    Use predicted line segments to compute scaling factors for the channel

    C and slope are the network's predictions, so they stay tensors here:
    taking float() of them severed the curve heads from autograd, leaving
    them trainable only through slope_sqr_diff.
    '''
    # clamp(x*steps - i, 0, 1) is the hinge that confines segment i to its
    # own interval: 0 below it, ramping across it, 1 above. Summed over the
    # segments with the slopes as weights, this is the piecewise linear
    # curve through the knots.
    #
    # Evaluated for all segments at once. As a Python loop this added 15
    # sequential nodes to the graph per curve, and with ten curves per
    # forward pass the backward pass was walking 150 of them.
    steps = torch.arange(slope.shape[-1], device=img.device, dtype=img.dtype)
    hinge = torch.clamp(
        img[..., channel_in].unsqueeze(-1)*curve_steps - steps, 0, 1)
    if batched:
        # img is (B, W, H, 3) and slope is (B, n_segments): give the slopes
        # two trailing singleton axes so each image meets its own curve.
        extra = (1,) * (hinge.dim() - 2)
        scale = (C[:, 0].view(-1, *extra)
                 + (hinge * slope.view(-1, *extra, slope.shape[-1])).sum(-1))
    else:
        scale = C[0] + (hinge * slope).sum(-1)

    img_copy = img.clone()
  
    img_copy[..., channel_out] = img[..., channel_out]*scale
    
    img_copy = torch.clamp(img_copy,0,1)
    
    return img_copy, slope_sqr_diff


def adjust_lab(img: torch.Tensor, L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adjusts the image in LAB space using the predicted curves

    :param img: Image tensor
    :param L: Predicited curve parameters for LAB channels
    :returns: adjust image, and regularisation parameter
    :rtype: Tensor, float

    """
    img = channels_last(img)

    img = img.contiguous()

    '''
    Extract predicted parameters for each L,a,b curve
    '''
    L1, L2, L3 = _raw_split(L, 3)

    slope_sqr_diff = torch.zeros(1, device=DEVICE)

    '''
    Apply the curve to the L channel 
    '''
    img_copy, slope_sqr_diff = _apply(
        img, L1, slope_sqr_diff, channel_in=0, channel_out=0)

    '''
    Now do the same for the a channel
    '''
    img_copy, slope_sqr_diff = _apply(
        img_copy, L2, slope_sqr_diff, channel_in=1, channel_out=1)

    '''
    Now do the same for the b channel
    '''
    img_copy, slope_sqr_diff = _apply(
        img_copy, L3, slope_sqr_diff, channel_in=2, channel_out=2)

    img = img_copy

    img = torch.nan_to_num(img)

    img = channels_last(img)
    img = img.contiguous()

    return img, slope_sqr_diff


def adjust_rgb(img: torch.Tensor, R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adjust the RGB channels of a RGB image using learnt curves

    :param img: image to be adjusted 
    :param S: predicted parameters of piecewise linear curves
    :returns: adjust image, regularisation term
    :rtype: Tensor, float

    """
    img = channels_last(img)
    img = img.contiguous()

    '''
    Extract the parameters of the three curves
    '''
    R1, R2, R3 = _raw_split(R, 3)

    '''
    Apply the curve to the R channel 
    '''
    slope_sqr_diff = torch.zeros(1, device=DEVICE)

    img_copy, slope_sqr_diff = _apply(
        img, R1, slope_sqr_diff, channel_in=0, channel_out=0)

    '''
    Apply the curve to the G channel 
    '''
    img_copy, slope_sqr_diff = _apply(
        img_copy, R2, slope_sqr_diff, channel_in=1, channel_out=1)

    '''
    Apply the curve to the B channel 
    '''
    img_copy, slope_sqr_diff = _apply(
        img_copy, R3, slope_sqr_diff, channel_in=2, channel_out=2)

    img = img_copy

    img = torch.nan_to_num(img)

    img = channels_last(img)
    img = img.contiguous()

    return img, slope_sqr_diff


def adjust_hsv(img: torch.Tensor, S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adjust the HSV channels of a HSV image using learnt curves

    :param img: image to be adjusted 
    :param S: predicted parameters of piecewise linear curves
    :returns: adjust image, regularisation term
    :rtype: Tensor, float

    """
    img = channels_last(img)
    img = img.contiguous()

    S1, S2, S3, S4 = _raw_split(S, 4)

    slope_sqr_diff = torch.zeros(1, device=DEVICE)

    '''
    Adjust Hue channel based on Hue using the predicted curve
    '''
    img_copy, slope_sqr_diff = _apply(
        img, S1, slope_sqr_diff, channel_in=0, channel_out=0)

    '''
    Adjust Saturation channel based on Hue using the predicted curve
    '''
    img_copy, slope_sqr_diff = _apply(
        img_copy, S2, slope_sqr_diff, channel_in=0, channel_out=1)
    
    '''
    Adjust Saturation channel based on Saturation using the predicted curve
    '''
    img_copy, slope_sqr_diff = _apply(
        img_copy, S3, slope_sqr_diff, channel_in=1, channel_out=1)

    '''
    Adjust Value channel based on Value using the predicted curve
    '''
    img_copy, slope_sqr_diff = _apply(
        img_copy, S4, slope_sqr_diff, channel_in=2, channel_out=2)

    img = img_copy

    img = torch.nan_to_num(img)

    img = channels_last(img)
    img = img.contiguous()
    
    return img, slope_sqr_diff
