# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""The CURL training loss.

Combines L1 in RGB, CIELab and HSV with a cosine term on RGB vectors, MS-SSIM
on the luminance channel, and the curve smoothness penalty.
"""
import math
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F

from colour import rgb_to_hsv, rgb_to_lab
from device import DEVICE


class CURLLoss(nn.Module):

    def __init__(self, ssim_window_size: int = 5, reg_weight: float = 1e-6):
        """Initialisation of the DeepLPF loss function

        :param ssim_window_size: size of averaging window for SSIM
        :returns: N/A
        :rtype: N/A

        """
        super(CURLLoss, self).__init__()
        self.ssim_window_size = ssim_window_size
        #: Weight on the curve smoothness penalty. At 1e-6 the term is about
        #: 0.001% of the loss, so it has no practical influence on the shape of
        #: the predicted curves.
        self.reg_weight = reg_weight

    def create_window(self, window_size, num_channel):
        """Window creation function for SSIM metric. Gaussian weights are applied to the window.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param window_size: size of the window to compute statistics
        :param num_channel: number of channels
        :returns: Tensor of shape Cx1xWindow_sizexWindow_size
        :rtype: Tensor

        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(
            num_channel, 1, window_size, window_size).contiguous()
        return window

    def gaussian(self, window_size, sigma):
        """
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
        :param window_size: size of the SSIM sampling window e.g. 11
        :param sigma: Gaussian variance
        :returns: 1xWindow_size Tensor of Gaussian weights
        :rtype: Tensor

        """
        gauss = torch.Tensor(
            [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def compute_ssim(self, img1, img2):
        """Computes the structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        (_, num_channel, _, _) = img1.size()
        window = self.create_window(self.ssim_window_size, num_channel)

        window = window.to(img1.device).type_as(img1)

        mu1 = F.conv2d(
            img1, window, padding=self.ssim_window_size // 2, groups=num_channel)
        mu2 = F.conv2d(
            img2, window, padding=self.ssim_window_size // 2, groups=num_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map1 = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
        ssim_map2 = ((mu1_sq.to(DEVICE) + mu2_sq.to(DEVICE) + C1) *
                     (sigma1_sq.to(DEVICE) + sigma2_sq.to(DEVICE) + C2))
        ssim_map = ssim_map1.to(DEVICE) / ssim_map2.to(DEVICE)

        v1 = 2.0 * sigma12.to(DEVICE) + C2
        v2 = sigma1_sq.to(DEVICE) + sigma2_sq.to(DEVICE) + C2
        cs = torch.mean(v1 / v2)

        return ssim_map.mean(), cs


    def compute_msssim(self, img1, img2):
        """Computes the multi scale structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        if img1.shape != img2.shape:
            raise RuntimeError('Input images must have the same shape (%s vs. %s).'
                               % (tuple(img1.shape), tuple(img2.shape)))
        if img1.ndim != 4:
            raise RuntimeError('Input images must have four dimensions, not %d'
                               % img1.ndim)

        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        ssims = []
        mcs = []
        for _ in range(levels):
            ssim, cs = self.compute_ssim(img1, img2)

            # Relu normalize (not compliant with original definition)
            ssims.append(ssim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        ssims = torch.stack(ssims)
        mcs = torch.stack(mcs)

        # Simple normalize (not compliant with original definition)
        # TODO: remove support for normalize == True (kept for backward support)
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = ssims ** weights

        # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
        # pow2[-1] is a scalar, so multiplying it into the four-element vector
        # before prod() raised it to the fourth power. MS-SSIM is the product
        # of the coarse-scale contrast terms times the finest-scale SSIM once.
        output = torch.prod(pow1[:-1]) * pow2[-1]
        return output

    def forward(self, predicted_img_batch, target_img_batch, gradient_regulariser):
        """Forward function for the CURL loss

        :param predicted_img_batch_high_res: 
        :param predicted_img_batch_high_res_rgb: 
        :param target_img_batch: Tensor of shape BxCxWxH
        :returns: value of loss function
        :rtype: float

        """

        # Computed over the whole batch at once. This was a Python loop over
        # the images, which serialised the loss and cancelled out any gain
        # from a batch size above one.
        predicted_img_lab = torch.clamp(
            rgb_to_lab(predicted_img_batch), 0, 1)
        target_img_lab = torch.clamp(
            rgb_to_lab(target_img_batch), 0, 1)

        predicted_img_hsv = torch.clamp(
            rgb_to_hsv(predicted_img_batch), 0, 1)
        target_img_hsv = torch.clamp(
            rgb_to_hsv(target_img_batch), 0, 1)

        # Hue is an angle, so it enters the loss through its sine and cosine
        # rather than as a number that wraps at 1.
        def _hsv_cartesian(hsv):
            hue = hsv[:, 0, :, :]*2*math.pi
            sat = hsv[:, 1, :, :]
            val = hsv[:, 2, :, :]
            return torch.stack(
                (val*sat*torch.cos(hue), val*sat*torch.sin(hue), val), 1)

        predicted_hsv_xy = _hsv_cartesian(predicted_img_hsv)
        target_hsv_xy = _hsv_cartesian(target_img_hsv)

        # MS-SSIM over the L channel, one call for the batch.
        ssim_value = self.compute_msssim(
            predicted_img_lab[:, 0:1, :, :], target_img_lab[:, 0:1, :, :])
        ssim_loss_value = 1.0 - ssim_value

        l1_loss_value = F.l1_loss(predicted_img_lab, target_img_lab)
        rgb_loss_value = F.l1_loss(predicted_img_batch, target_img_batch)
        hsv_loss_value = F.l1_loss(predicted_hsv_xy, target_hsv_xy)

        cosine_rgb_loss_value = 1 - torch.mean(
            torch.nn.functional.cosine_similarity(
                predicted_img_batch, target_img_batch, dim=1))

        curl_loss = (rgb_loss_value + cosine_rgb_loss_value + l1_loss_value +
                     hsv_loss_value + 10*ssim_loss_value
                     + self.reg_weight*gradient_regulariser)/6

        return curl_loss
