# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Authors: Sean Moran (sean.j.moran@gmail.com),
         Pierre Marza (pierre.marza@gmail.com)

Code-to-paper map (section / equation numbers refer to the arXiv version):

* ``unet.UNetModel``             backbone, Sec. 3.1 (its first three output channels
                                 are the backbone-enhanced image Y1)
* ``cubic.CubicFilter``          polynomial ("cubic-20") filter, Sec. 3.2.4, Eq. 6 -> Y2
* ``graduated.GraduatedFilter``  graduated filter, Sec. 3.2.2, Eqs. 1-3, fused per Eq. 7 -> s_g
* ``elliptical.EllipticalFilter`` elliptical filter, Sec. 3.2.3, Eq. 4, fused per Eq. 7 -> s_e
* ``DeepLPFParameterPrediction`` fusion S = 1 + (s_g - 1) + (s_e - 1), Y3 = S * Y2,
                                 Y = Y3 + Y1, Sec. 3.1
* ``losses.DeepLPFLoss``         training loss, Sec. 3.4, Eq. 8

The filter heads live in :mod:`cubic`, :mod:`graduated` and :mod:`elliptical`,
the conv blocks they are built from in :mod:`blocks`, and the loss in
:mod:`losses`. They are re-exported here so that ``model.CubicFilter`` and
``model.DeepLPFLoss`` keep resolving, as they did when every class lived in
this file. The module hierarchy of the network is unchanged, so state dicts
saved before this split load unchanged.
'''
import torch
import torch.nn as nn

import unet
# Re-exported so that model.CubicFilter and model.DeepLPFLoss resolve, as
# they did when every class lived in this file.
from blocks import Block, ConvBlock, GlobalPoolingBlock, MaxPoolBlock  # noqa: F401
from filters import (BinaryLayer, CubicFilter, EllipticalFilter,  # noqa: F401
                     GraduatedFilter, SignSTE)
from losses import DeepLPFLoss  # noqa: F401


class ColourHead(nn.Module):
    """Global colour mixer and per-channel tone curve.

    Every parametric filter is diagonal: CubicFilter's coefficients
    are reshaped to (B, 3, 1, 1) and every term of Eq. 6 multiplies that same
    channel, and the graduated and elliptical branches only apply per-channel
    gains. So no head can express a cross-channel operation - white balance,
    saturation, a hue shift - although those are most of what the expert
    retouch consists of. The U-Net can approximate them locally, which is why
    this is a gap in the filter bank rather than in the model as a whole.

    Two operators, both global and both interpretable:

    ``M Y1 + b``
        a 3x3 mixer and offset, the cross-channel term the model lacks. This
        is the Calibration/white-balance panel of a raw converter.
    ``i + sum_k w_ck relu(i - t_k)``
        a piecewise-linear tone curve per channel with fixed knots, so the
        slope on segment j is ``1 + sum_{k<=j} w_k`` and the learned curve
        plots directly. Hinges rather than a gather, so the shape is static
        and the op is safe to capture in a CUDA graph.

    Neither operator clamps. Y1 is not bounded to [0, 1] where this runs - the
    cubic filter clamps only after adding its residual - so a clamp here would
    alter the image even with a zero prediction, and the head would no longer
    start from the identity.

    Both are zero-initialised: at initialisation ``M = I``, ``b = 0``, ``w = 0``
    and the head is exactly the identity, so turning the feature on does not
    change the starting model. The layer is constructed after every other
    module so that the RNG stream the existing parameters draw from is
    untouched, and a run with this feature enabled starts from the same weights
    as one without it at the same seed.

    """

    def __init__(self, in_channels=64, knots=16):
        """Initialise the colour head.

        :param in_channels: channels of the pooled (features, image) input
        :param knots: tone-curve knots; 0 leaves the mixer alone
        :returns: N/A
        :rtype: N/A

        """
        super(ColourHead, self).__init__()
        self.knots = int(knots)
        self.fc = torch.nn.Linear(in_channels, 12 + 3 * self.knots)
        # Identity at initialisation, and still trainable: the gradient with
        # respect to a zero weight is the input times the upstream gradient,
        # which is not zero.
        torch.nn.init.zeros_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)
        self.register_buffer('eye', torch.eye(3).unsqueeze(0))
        if self.knots:
            self.register_buffer(
                'thresholds',
                (torch.arange(self.knots, dtype=torch.float32) / self.knots)
                .view(1, 1, self.knots, 1, 1))

    def forward(self, context, img):
        """Apply the mixer and curve to an image.

        :param context: (B, C, H, W) features concatenated with the image,
                        global-average-pooled to drive the prediction
        :param img: (B, 3, H, W) image to transform
        :returns: transformed image, (B, 3, H, W)
        :rtype: Tensor

        """
        params = self.fc(context.mean(dim=(2, 3)))

        # Mixer: M = I + dM, so a zero prediction is the identity.
        mixer = params[:, 0:9].view(-1, 3, 3) + self.eye
        offset = params[:, 9:12].view(-1, 3, 1, 1)
        out = torch.einsum('bij,bjhw->bihw', mixer, img) + offset

        if self.knots:
            weights = params[:, 12:].view(-1, 3, self.knots, 1, 1)
            hinges = torch.clamp(out.unsqueeze(2) - self.thresholds, min=0)
            out = out + (weights * hinges).sum(dim=2)

        return out


class DeepLPFParameterPrediction(nn.Module):
    import torch.nn.functional as F

    def __init__(self, num_in_channels=64, num_out_channels=64,
                 learn_filter_count=False, colour_knots=None):
        """Initialisation function

        :param num_in_channels:  Number of input feature maps
        :param num_out_channels: Number of output feature maps
        :param learn_filter_count: predict a gate per filter instance
        :param colour_knots: knots in the colour head's tone curve, or None to
                             leave the colour head out entirely
        :returns: N/A
        :rtype: N/A

        """
        super(DeepLPFParameterPrediction, self).__init__()
        self.cubic_filter = CubicFilter()
        self.graduated_filter = GraduatedFilter(learn_filter_count=learn_filter_count)
        self.elliptical_filter = EllipticalFilter(learn_filter_count=learn_filter_count)
        self.learn_filter_count = learn_filter_count
        # Constructed after every other module on purpose: see ColourHead.
        self.use_colour = colour_knots is not None
        if self.use_colour:
            self.colour_head = ColourHead(64, colour_knots)
      

    def forward(self, x):
        """DeepLPF combined architecture fusing cubic, graduated and elliptical filters

        :param x: forward the data Tensor x through the network
        :returns: Tensor representing the predicted image batch of shape BxCxWxH
        :rtype: Tensor

        """
        x = x.contiguous()  # remove memory holes

        feat = x[:, 3:64, :, :]  # C' = C - 3 backbone features
        img = x[:, 0:3, :, :]    # Y1: backbone-enhanced image

        # `--colour_head`: the cross-channel operation no filter head can
        # express, applied to Y1 before the cubic filter so every downstream
        # branch - including the parameter predictors, which read the image -
        # sees the corrected colour. Identity at initialisation.
        if self.use_colour:
            img = self.colour_head(x, img)

        # Each branch's parameter predictor consumes cat(feat, image) resized
        # to 300x300. The resize is bilinear and per channel, so
        # upsample(cat(a, b)) == cat(upsample(a), upsample(b)) exactly; the
        # feature part is shared by all three branches, and the graduated and
        # elliptical branches share the whole input. Resize each part once
        # instead of five times.
        upsample = self.cubic_filter.upsample
        feat_up = upsample(feat)

        # Single-stream path: Y2 = polynomial filter applied to Y1
        img_cubic = self.cubic_filter.mask_from_input(
            torch.cat((feat_up, upsample(img)), 1), img)

        # Two-stream path: graduated and elliptical scaling maps estimated from Y2
        feat_img_cubic_up = torch.cat((feat_up, upsample(img_cubic)), 1)
        mask_scale_graduated = self.graduated_filter.mask_from_input(
            feat_img_cubic_up, img_cubic)
        mask_scale_elliptical = self.elliptical_filter.mask_from_input(
            feat_img_cubic_up, img_cubic)

        mask_scale_fuse = torch.clamp(
            1.0 + (mask_scale_graduated - 1.0) + (mask_scale_elliptical - 1.0), 0, 2)

        # Mean gate over both branches' instances, in [0, 1]. Penalising this
        # penalises the expected number of active filters; the training step
        # adds gate_weight * gate_penalty to the loss.
        if self.learn_filter_count:
            self.gate_penalty = torch.cat(
                (self.graduated_filter.gates, self.elliptical_filter.gates), 1).mean()

        img_fuse = torch.clamp(img_cubic*mask_scale_fuse, 0, 1)
        
        img = torch.clamp(img_fuse+img, 0, 1)
        
        return img


class DeepLPFNet(nn.Module):

    def __init__(self, learn_filter_count=False, colour_knots=None):
        """Initialisation function

        :param learn_filter_count: predict a gate per filter instance, so the
                                   network learns how many of the three
                                   instances per branch an image needs
        :param colour_knots: knots in the per-channel tone curve of the colour
                             head; None leaves the colour head out, 0 keeps the
                             mixer without a curve
        :returns: initialises parameters of the neural network
        :rtype: N/A

        """
        super(DeepLPFNet, self).__init__()
        self.learn_filter_count = learn_filter_count
        self.backbonenet = unet.UNetModel()
        self.deeplpfnet = DeepLPFParameterPrediction(
            learn_filter_count=learn_filter_count, colour_knots=colour_knots)
        
    def forward(self, img):
        """Neural network forward function

        :param img: forward the data img through the network
        :returns: residual image
        :rtype: numpy ndarray

        """
        feat = self.backbonenet(img)
        img = self.deeplpfnet(feat)

        # Surface the gate penalty of the last forward pass so the training
        # step can add it to the loss without changing this signature.
        if self.learn_filter_count:
            self.gate_penalty = self.deeplpfnet.gate_penalty
        
        return img
