# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""Colour-space conversions.

Each function takes a CHW image or a BCHW batch and returns the same rank.
Lab and HSV come back normalised to [0, 1] on every channel."""
import torch

from device import DEVICE
from tensors import channels_last


def rgb_to_lab(img: torch.Tensor, is_training: bool = True) -> torch.Tensor:
    """ PyTorch implementation of RGB to LAB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
    :param img: image to be adjusted
    :returns: adjusted image
    :rtype: Tensor

    """
    img = channels_last(img)
    shape = img.shape
    img = img.contiguous()
    img = img.view(-1, 3)

    img = (img / 12.92) * img.le(0.04045).float() + (((torch.clamp(img,
                                                                   min=0.0001) + 0.055) / 1.055) ** 2.4) * img.gt(0.04045).float()

    rgb_to_xyz = torch.FloatTensor([  # X        Y          Z
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169,
         0.950227],  # B
    ]).to(DEVICE)

    img = torch.matmul(img, rgb_to_xyz)
    img = torch.mul(img, torch.FloatTensor(
        [1/0.950456, 1.0, 1/1.088754]).to(DEVICE))

    epsilon = 6/29

    img = ((img / (3.0 * epsilon**2) + 4.0/29.0) * img.le(epsilon**3).float()) + \
        (torch.clamp(img, min=0.0001) **
         (1.0/3.0) * img.gt(epsilon**3).float())

    fxfyfz_to_lab = torch.FloatTensor([[0.0,  500.0,    0.0],  # fx
                                                # fy
                                                [116.0, -500.0,  200.0],
                                                # fz
                                                [0.0,    0.0, -200.0],
                                                ]).to(DEVICE)

    img = torch.matmul(img, fxfyfz_to_lab) + torch.FloatTensor([-16.0, 0.0, 0.0]).to(DEVICE)

    img = img.view(shape)
    img = channels_last(img)

    '''
    L_chan: black and white with input range [0, 100]
    a_chan/b_chan: color channels with input range ~[-110, 110], not exact 
    [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]
    '''
    img[..., 0, :, :] = img[..., 0, :, :]/100
    img[..., 1, :, :] = (img[..., 1, :, :]/110 + 1)/2
    img[..., 2, :, :] = (img[..., 2, :, :]/110 + 1)/2

    img = torch.nan_to_num(img)

    img = img.contiguous()

    return img.to(DEVICE)


def lab_to_rgb(img: torch.Tensor, is_training: bool = True) -> torch.Tensor:
    """ PyTorch implementation of LAB to RGB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
    :param img: image to be adjusted
    :returns: adjusted image
    :rtype: Tensor
    """                
    img = channels_last(img)
    shape = img.shape
    img = img.contiguous()
    img = img.view(-1, 3)
    img_copy = img.clone()

    img_copy[..., 0] = img[..., 0] * 100
    img_copy[..., 1] = ((img[..., 1] * 2)-1)*110
    img_copy[..., 2] = ((img[..., 2] * 2)-1)*110

    img = img_copy.to(DEVICE)

    lab_to_fxfyfz = torch.FloatTensor([  # X Y Z
        [1/116.0, 1/116.0, 1/116.0],  # R
        [1/500.0, 0, 0],  # G
        [0, 0, -1/200.0],  # B
    ]).to(DEVICE)

    img = torch.matmul(
        img + torch.tensor([16.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE), lab_to_fxfyfz)

    epsilon = 6.0/29.0

    img = (((3.0 * epsilon**2 * (img-4.0/29.0)) * img.le(epsilon).float()) +
           ((torch.clamp(img, min=0.0001)**3.0) * img.gt(epsilon).float()))

    # denormalize for D65 white point
    img = torch.mul(img, torch.tensor([0.950456, 1.0, 1.088754], dtype=torch.float32, device=DEVICE))

    xyz_to_rgb = torch.FloatTensor([  # X Y Z
        [3.2404542, -0.9692660,  0.0556434],  # R
        [-1.5371385,  1.8760108, -0.2040259],  # G
        [-0.4985314,  0.0415560,  1.0572252],  # B
    ]).to(DEVICE)

    img = torch.matmul(img, xyz_to_rgb)

    img = (img * 12.92 * img.le(0.0031308).float()) + ((torch.clamp(img,
                                                                    min=0.0001) ** (1/2.4) * 1.055) - 0.055) * img.gt(0.0031308).float()

    img = img.view(shape)
    img = channels_last(img)

    img = img.contiguous()
    img = torch.nan_to_num(img)
    
    return img


def rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    """Converts an RGB image to HSV
    PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

    :param img: RGB image
    :returns: HSV image
    :rtype: Tensor

    """
    img = torch.clamp(img, 1e-9, 1)

    img = channels_last(img)
    img = img.contiguous()

    mx, _ = img.max(-1)
    mn, _ = img.min(-1)
    df = (mx - mn) + 1e-10     # leading dims of `shape`, no reshaping

    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    # Pick exactly one branch. When two channels tie for the maximum -- pure
    # yellow, cyan, magenta, or any pixel with two equal channels -- more than
    # one of r.eq(mx), g.eq(mx), b.eq(mx) is true, and summing the branches adds
    # their contributions instead of selecting one: yellow takes the red and the
    # green branch, lands on hue 120 degrees, and comes back out of hsv_to_rgb
    # as green.
    is_r = r.eq(mx)
    is_g = g.eq(mx) & ~is_r
    is_b = b.eq(mx) & ~is_r & ~is_g
    hue = ((g-b)/df)*is_r.float() \
        + (2.0+(b-r)/df)*is_g.float() \
        + (4.0+(r-g)/df)*is_b.float()
    hue = hue*60.0
    hue = torch.where(hue < 0, hue+360, hue)/360

    sat = torch.where(mx != 0, df/mx, torch.zeros_like(mx))

    img = torch.stack((hue, sat, mx), dim=-1)
    img = torch.nan_to_num(img)

    img = channels_last(img)
    img = torch.clamp(img, 1e-9, 1)

    return img


def hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
    """Converts a HSV image to RGB
    PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

    :param img: HSV image
    :returns: RGB image
    :rtype: Tensor

    """
    img=torch.clamp(img,0,1)
    img = channels_last(img)
    
    m1 = 0
    m2 = (img[..., 2]*(1-img[..., 1])-img[..., 2])/60
    m3 = 0
    m4 = -1*m2
    m5 = 0

    r = img[..., 2]+torch.clamp(img[..., 0]*360-0, 0, 60)*m1+torch.clamp(img[..., 0]*360-60, 0, 60)*m2+torch.clamp(
        img[..., 0]*360-120, 0, 120)*m3+torch.clamp(img[..., 0]*360-240, 0, 60)*m4+torch.clamp(img[..., 0]*360-300, 0, 60)*m5

    m1 = (img[..., 2]-img[..., 2]*(1-img[..., 1]))/60
    m2 = 0
    m3 = -1*m1
    m4 = 0

    g = img[..., 2]*(1-img[..., 1])+torch.clamp(img[..., 0]*360-0, 0, 60)*m1+torch.clamp(img[..., 0]*360-60,
                                                                                            0, 120)*m2+torch.clamp(img[..., 0]*360-180, 0, 60)*m3+torch.clamp(img[..., 0]*360-240, 0, 120)*m4

    m1 = 0
    m2 = (img[..., 2]-img[..., 2]*(1-img[..., 1]))/60
    m3 = 0
    m4 = -1*m2

    b = img[..., 2]*(1-img[..., 1])+torch.clamp(img[..., 0]*360-0, 0, 120)*m1+torch.clamp(img[..., 0]*360 -
                                                                                             120, 0, 60)*m2+torch.clamp(img[..., 0]*360-180, 0, 120)*m3+torch.clamp(img[..., 0]*360-300, 0, 60)*m4

    img = torch.stack((r, g, b), -1)
    img = torch.nan_to_num(img)

    img = channels_last(img)
    img = img.contiguous()
    img = torch.clamp(img, 0, 1)

    return img
