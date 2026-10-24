# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""The elliptical filter of Sec. 3.2.3, Eq. 4.

It predicts ellipses - centre, semi-axes and rotation - and scales the image
within each one, giving the radial and vignette-style adjustments.
"""
import math

import torch
import torch.nn as nn

from blocks import ConvBlock, GlobalPoolingBlock, MaxPoolBlock
from filtercommon import GATES_PER_BRANCH, _apply_gates, _coord_grids


class EllipticalFilter(nn.Module):
    """Elliptical-filter branch of DeepLPF (paper Sec. 3.2.3, Eq. 4; Sec. 3.3, Eq. 7).

    Regresses ``E = 24`` values: for each of three ellipse instances the centre
    ``(h, k)``, semi-axes ``(a, b)``, rotation ``theta`` and three per-channel
    scaling factors ``s_e^R, s_e^G, s_e^B``. Inside an ellipse the scaling is
    ``s_e`` at the centre and decays linearly to 1 at the boundary (the code's
    linear-in-radius form of Eq. 4); outside it is 1 (no change). Each instance
    yields a (B, 3, H, W) map; the three are fused by element-wise
    multiplication (Eq. 7) into ``s_e``. Scalings are bounded to [0, max_scale = 2].
    """

    def __init__(self, num_in_channels=64, num_out_channels=64,
                 learn_filter_count=False):
        """Build the elliptical-filter branch.

        :param num_in_channels: number of input feature-map channels
        :param num_out_channels: number of channels used by the conv stack
        :param learn_filter_count: predict one gate per filter instance
        :returns: N/A
        :rtype: N/A

        """
        super(EllipticalFilter, self).__init__()
        self.learn_filter_count = learn_filter_count

        self.elliptical_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.elliptical_layer2 = MaxPoolBlock()
        self.elliptical_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer4 = MaxPoolBlock()
        self.elliptical_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer6 = MaxPoolBlock()
        self.elliptical_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer8 = GlobalPoolingBlock()
        # 24 filter parameters, plus one gate per instance when the filter
        # count is learned.
        self.fc_elliptical = torch.nn.Linear(
            num_out_channels, 24 + (GATES_PER_BRANCH if learn_filter_count else 0))
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear',align_corners=False)
        self.dropout = nn.Dropout(0.5)

    def tanh01(self, x):
        """Adjust Tanh to return values between 0 and 1

        :param x: Tensor arbitrary range
        :returns: Tensor between 0 and 1
        :rtype: tensor

        """
        tanh = nn.Tanh()
        return 0.5 * (tanh(x) + 1)

    def where(self, cond, x_1, x_2):
        """Differentiable where function to compare two Tensors

        :param cond: condition e.g. <
        :param x_1: Tensor 1
        :param x_2: Tensor 2
        :returns: Boolean comparison result
        :rtype: Tensor

        """
        cond = cond.float()
        return (cond * x_1) + ((1 - cond) * x_2)

    def get_mask(self, x_axis, y_axis, shift_x=0, shift_y=0, semi_axis_x=1, semi_axis_y=1, alpha=0,
                 scale_factor=2, max_scale=2, eps=1e-7, radius=1):
        """Gets the elliptical scaling mask according to the equation of a
        rotated ellipse

        :param x_axis: normalised x-coordinate grid
        :param y_axis: normalised y-coordinate grid
        :param shift_x: ellipse centre x-coordinate
        :param shift_y: ellipse centre y-coordinate
        :param semi_axis_x: ellipse semi-axis along x
        :param semi_axis_y: ellipse semi-axis along y
        :param alpha: rotation angle of the ellipse
        :param scale_factor: peak scaling applied at the ellipse centre
        :param max_scale: maximum permitted scaling factor
        :param eps: small constant to avoid division by zero
        :param radius: ellipse radius at the current angle
        :returns: scaling mask
        :rtype: Tensor

        """
        # Rotated-ellipse membership test (the bracketed term of Eq. 4):
        # [(x-h)cos t + (y-k)sin t]^2 / a^2 + [(x-h)sin t - (y-k)cos t]^2 / b^2 < 1
        ellipse_equation_part1 = (((x_axis - shift_x)*torch.cos(alpha) + (y_axis - shift_y)*torch.sin(alpha)) ** 2) / ((semi_axis_x)**2)
        ellipse_equation_part2 = (((x_axis - shift_x)*torch.sin(alpha) - (y_axis - shift_y)*torch.cos(alpha)) ** 2) / ((semi_axis_y)**2)

        # Inside: scale_factor at the centre, decaying linearly with the distance
        # from the centre to reach 1 at the ellipse boundary (`radius` is the
        # boundary distance along the pixel's direction). Outside: 1 (no change).
        mask_scale = self.where(ellipse_equation_part1+ellipse_equation_part2 < 1,
                                (torch.sqrt((x_axis - shift_x) ** 2 + (y_axis - shift_y) ** 2 + eps) * (1 - scale_factor)) / radius + scale_factor, 1)

        # Returns a (B, H, W) per-image, per-ellipse scaling map.
        mask_scale = torch.clamp(mask_scale, 0, max_scale)

        return mask_scale

    def get_elliptical_mask(self, feat, img):
        """Gets the elliptical scaling mask according to the equation of a
        rotated ellipse

        :param feat: features from the backbone
        :param img: image
        :returns: elliptical adjustment maps for each channel
        :rtype: Tensor

        """

        feat_elliptical = torch.cat((feat, img), 1)
        feat_elliptical = self.upsample(feat_elliptical)
        return self.mask_from_input(feat_elliptical, img)

    def mask_from_input(self, feat_elliptical, img):
        """As :meth:`get_elliptical_mask`, given the already resized 300x300 input.

        :param feat_elliptical: (B, 64, 300, 300) resized concatenation of features and image
        :param img: image the filter is applied to, (B, 3, H, W)
        :returns: elliptical adjustment maps for each channel, (B, 3, H, W)
        :rtype: Tensor

        """
        # The two eps parameters are used to avoid numerical issues in the learning
        eps2 = 1e-7
        eps1 = 1e-10

        # max_scale is the maximum an ellipse can scale the image R,G,B values by
        max_scale = 2

        # Parameter prediction (Sec. 3.2.1): image + backbone features -> 24 values
        x = self.elliptical_layer1(feat_elliptical)
        x = self.elliptical_layer2(x)
        x = self.elliptical_layer3(x)
        x = self.elliptical_layer4(x)
        x = self.elliptical_layer5(x)
        x = self.elliptical_layer6(x)
        x = self.elliptical_layer7(x)
        x = self.elliptical_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        G = self.fc_elliptical(x)

        # The next code implements a rotated ellipse according to:
        # https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate
        
        # Normalised coordinates for x and y-axes, we instantiate the ellipses in these coordinates
        x_axis, y_axis = _coord_grids(img.shape[2], img.shape[3], img.device)

        # G has shape (B, 24). The three ellipse instances are evaluated
        # together along a leading "instance" dim: each per-instance parameter
        # is a (B, 3) vector reshaped to (B, 3, 1, 1, 1) so it broadcasts over
        # the (H, W) coordinate grids, and the nine (instance, channel) masks
        # come out of one get_mask call as (B, 3, 3, H, W). This replaced three
        # copies of every line and nine get_mask calls; the per-element
        # arithmetic is unchanged.
        # Layout: [0:3] h, [3:6] k, [6:9] a, [9:12] b, [12:15] theta,
        # [15:24] per-channel scale factors s_e (three instances x R,G,B).
        # Centre of ellipse, x-coordinate (h in Eq. 4)
        x_coord = (self.tanh01(G[:, 0:3]) + eps1).view(-1, 3, 1, 1, 1)

        # Centre of ellipse, y-coordinate (k in Eq. 4)
        y_coord = (self.tanh01(G[:, 3:6]) + eps1).view(-1, 3, 1, 1, 1)

        # Semi-major axis a
        a = (self.tanh01(G[:, 6:9]) + eps1).view(-1, 3, 1, 1, 1)

        # Semi-minor axis b
        b = (self.tanh01(G[:, 9:12]) + eps1).view(-1, 3, 1, 1, 1)

        # Rotation angle theta in [0, pi]
        A = (self.tanh01(G[:, 12:15]) * math.pi + eps1).view(-1, 3, 1, 1, 1)

        # Per-channel scale factors s_e in [0, max_scale] for the three
        # instances: (B, instance, channel, 1, 1)
        scale = (self.tanh01(G[:, 15:24]) * max_scale + eps1).view(-1, 3, 3, 1, 1)

        # Polar angle of every pixel about the ellipse centre, measured from the
        # y semi-axis and offset by the ellipse rotation. The clamp keeps acos
        # inside its domain so its gradient stays finite. (B, 3, 1, H, W)
        angle = torch.acos(torch.clamp((y_axis-y_coord) /
            (torch.sqrt((x_axis-x_coord)**2 + (y_axis-y_coord)**2 + eps1)), -1+eps2, 1-eps2))-A

        # Distance from the centre to the ellipse boundary along each pixel's angle,
        # r(phi) = a b / sqrt(a^2 sin^2 phi + b^2 cos^2 phi); this is the normaliser
        # for the linear decay in get_mask.
        # https://math.stackexchange.com/questions/432902/how-to-get-the-radius-of-an-ellipse-at-a-specific-angle-by-knowing-its-semi-majo
        radius = (a*b)/torch.sqrt((a**2)*(torch.sin(angle)**2)+(b**2)*(torch.cos(angle)**2) + eps1)

        # Every instance's three channels share one ellipse.
        semi_axis_y = b

        # Every instance: one ellipse geometry, three per-channel (R, G, B)
        # scalings -> (B, instance, channel, H, W)
        mask_scale = self.get_mask(x_axis, y_axis,
                                   shift_x=x_coord, shift_y=y_coord, semi_axis_x=a, semi_axis_y=semi_axis_y,
                                   alpha=angle, scale_factor=scale, radius=radius)
        mask_scale_rad = torch.clamp(mask_scale, 0, max_scale)

        # `--learn_filter_count`: see GraduatedFilter.mask_from_input.
        if self.learn_filter_count:
            self.gates = self.tanh01(G[:, 24:27])
            mask_scale_rad = _apply_gates(mask_scale_rad, self.gates)

        # Fuse the three instances by element-wise multiplication: s_e = prod_i s_ei (Eq. 7)
        mask_scale_elliptical = torch.clamp(
            mask_scale_rad[:, 0] * mask_scale_rad[:, 1] * mask_scale_rad[:, 2], 0, max_scale)

        return mask_scale_elliptical



