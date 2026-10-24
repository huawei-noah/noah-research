# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""The training loss of Sec. 3.4, Eq. 8, and the MS-SSIM it is built on.

The loss is an L1 term in CIELab plus a weighted MS-SSIM term on the lightness
channel, Eq. 8 of the paper.
"""
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import ImageProcessing

#: Weight on the MS-SSIM term of Eq. 8 (Sec. 4.1).
MSSSIM_WEIGHT = 1e-3


class DeepLPFLoss(nn.Module):
    """DeepLPF training loss (paper Sec. 3.4, Eq. 8).

    ``L = w_lab * ||Lab(Y_hat) - Lab(Y)||_1 + w_msssim * (1 - MS-SSIM(L(Y_hat), L(Y)))``

    with the L1 term taken over all three CIELab channels (rescaled to [0, 1]
    by :func:`util.ImageProcessing.rgb_to_lab`) and the MS-SSIM term on the L
    (lightness) channel only. The weights are hard-coded to the values the
    paper reports for MIT-Adobe-5K-DPE: ``w_lab = 1``, ``w_msssim = 1e-3``.
    Each image in the batch is scored separately and the terms are averaged
    over the batch.
    """

    def __init__(self, ssim_window_size=5):
        """Initialisation of the DeepLPF loss function

        :param ssim_window_size: size of averaging window for SSIM
        :returns: N/A
        :rtype: N/A

        """
        super(DeepLPFLoss, self).__init__()
        self.ssim_window_size = ssim_window_size
        # Per-(device, dtype) caches of the constant tensors the loss uses, so
        # they are copied to the device once rather than on every call.
        self._window_cache = {}
        self._weights_cache = {}

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
        key = (num_channel, img1.device, img1.dtype)
        window = self._window_cache.get(key)
        if window is None:
            # Move the SSIM window to the same device and dtype as the input
            window = self.create_window(self.ssim_window_size, num_channel).type_as(img1)
            self._window_cache[key] = window

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
        ssim_map2 = ((mu1_sq + mu2_sq + C1) *
                     (sigma1_sq + sigma2_sq + C2))
        ssim_map = ssim_map1 / ssim_map2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
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
        if img1.shape[2]!=img2.shape[2]:
                img1=img1.transpose(2,3)

        if img1.shape != img2.shape:
            raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
        if img1.ndim != 4:
            raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.ndim)

        device = img1.device
        weights = self._weights_cache.get(device)
        if weights is None:
            weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
            self._weights_cache[device] = weights
        levels = weights.size()[0]
        ssims = []
        mcs = []
        for _ in range(levels):
            ssim, cs = self.compute_ssim(img1, img2)
            ssims.append(ssim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        ssims = torch.stack(ssims)
        mcs = torch.stack(mcs)

        # Map SSIM / contrast-sensitivity from [-1, 1] to [0, 1] before raising
        # to the fractional per-scale weights (avoids NaN from a negative base).
        # This "simple normalisation" departs from the original Wang et al.
        # definition and is kept because the released models were trained with it.
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = ssims ** weights

        # Combine the per-scale contrast-sensitivity terms with the finest-scale SSIM
        # (cf. the Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/).
        #
        # MS-SSIM is prod_{j<J} cs_j^{w_j} times a single ssim_J^{w_J}, so
        # pow2[-1] is a factor of the product, not a factor of each term.
        #
        # Written as explicit products rather than torch.prod: prod's CUDA
        # backward counts the zeros in its input with a host sync, which is
        # illegal inside a CUDA graph capture (--cuda_graphs). Bitwise equal
        # to prod on CPU; on CUDA and MPS within 1 ulp of the MS-SSIM factor.
        p = pow1[:-1]
        return p[0] * p[1] * p[2] * p[3] * pow2[-1]

    def forward(self, predicted_img_batch, target_img_batch):
        """Forward function for the DeepLPF loss

        :param predicted_img_batch: predicted image Tensor of shape BxCxHxW
        :param target_img_batch: ground-truth image Tensor of shape BxCxHxW
        :returns: value of loss function
        :rtype: Tensor

        """
        if predicted_img_batch.shape[2]!=target_img_batch.shape[2]:
                target_img_batch=target_img_batch.transpose(2,3)

        num_images = target_img_batch.shape[0]

        # Accumulate the loss on the same device/dtype as the predictions
        ssim_loss_value = predicted_img_batch.new_zeros((1, 1))
        l1_loss_value = predicted_img_batch.new_zeros((1, 1))

        for i in range(0, num_images):

            target_img = target_img_batch[i, :, :, :]
            predicted_img = predicted_img_batch[i, :, :, :]

            # Lab(.) of Eq. 8: CIELab with every channel rescaled to [0, 1]
            predicted_img_lab = ImageProcessing.rgb_to_lab(
                predicted_img.squeeze(0))
            target_img_lab = ImageProcessing.rgb_to_lab(target_img.squeeze(0))

            # L(.) of Eq. 8: the lightness channel, as a 1x1xHxW batch for MS-SSIM
            target_img_L_ssim = target_img_lab[0, :, :].unsqueeze(0)
            predicted_img_L_ssim = predicted_img_lab[0, :, :].unsqueeze(0)
            target_img_L_ssim = target_img_L_ssim.unsqueeze(0)
            predicted_img_L_ssim = predicted_img_L_ssim.unsqueeze(0)

            ssim_value = self.compute_msssim(
                predicted_img_L_ssim, target_img_L_ssim)
            ssim_loss_value += (1.0 - ssim_value)

            l1_loss_value += F.l1_loss(predicted_img_lab, target_img_lab)

        l1_loss_value = l1_loss_value/num_images
        ssim_loss_value = ssim_loss_value/num_images
        # Eq. 8 with w_lab = 1. The paper's MIT-Adobe-5K-DPE setting is
        # w_msssim = 1e-3, at which the structural term contributes about 0.07%
        # of the L1 term's gradient - measured in
        # tests/test_loss_terms_contribute.py - so what trains the model is L1
        # in Lab space.
        deeplpf_loss = l1_loss_value + MSSSIM_WEIGHT*ssim_loss_value
        return deeplpf_loss


