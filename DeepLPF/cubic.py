# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""The polynomial ("cubic-20") filter of Sec. 3.2.4, Eq. 6.

It predicts the coefficients of a cubic polynomial in pixel intensity and image
coordinates, which is the global tone and colour curve applied to Y1.
"""
import torch
import torch.nn as nn

from blocks import ConvBlock, GlobalPoolingBlock, MaxPoolBlock
from filtercommon import _coord_grid_powers


class CubicFilter(nn.Module):
    """Cubic-polynomial ("cubic-20") filter branch of DeepLPF (paper Sec. 3.2.4, Eq. 6).

    Regresses 60 coefficients (20 per RGB channel, ``{A..T}`` in Eq. 6) from the
    backbone features and evaluates the cubic polynomial ``f(x, y, i)`` in the
    normalised pixel coordinates ``(x, y)`` and the channel intensity ``i`` at
    every pixel. Note the code applies it as a residual, ``i' = clamp(i + f)``,
    whereas Eq. 6 writes ``i' = f``.

    The parameter-prediction sub-network follows Sec. 3.2.1: conv/max-pool
    stages, global average pooling (so it is resolution agnostic), dropout 0.5,
    and a fully-connected regressor. The input is first resized to 300x300.
    """

    def __init__(self, num_in_channels=64, num_out_channels=64):
        """Build the cubic-filter branch.

        :param num_in_channels: number of input feature-map channels
        :param num_out_channels: number of channels used by the conv stack
        :returns: N/A
        :rtype: N/A

        """
        super(CubicFilter, self).__init__()

        self.cubic_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.cubic_layer2 = MaxPoolBlock()
        self.cubic_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer4 = MaxPoolBlock()
        self.cubic_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer6 = MaxPoolBlock()
        self.cubic_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer8 = GlobalPoolingBlock()
        self.fc_cubic = torch.nn.Linear(
            num_out_channels, 60)  # cubic
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear',align_corners=False)
        self.dropout = nn.Dropout(0.5)

    def get_cubic_mask(self, feat, img):
        """Cubic filter definition

        :param feat: feature map
        :param img:  image
        :returns: cubic scaling map
        :rtype: Tensor

        """
        # Parameter prediction (Sec. 3.2.1): image + backbone features -> 60 coefficients
        feat_cubic = torch.cat((feat, img), 1)
        feat_cubic = self.upsample(feat_cubic)
        return self.mask_from_input(feat_cubic, img)

    def mask_from_input(self, feat_cubic, img):
        """As :meth:`get_cubic_mask`, given the already resized 300x300 input.

        :param feat_cubic: (B, 64, 300, 300) resized concatenation of features and image
        :param img: image the filter is applied to, (B, 3, H, W)
        :returns: cubic-filtered image, (B, 3, H, W)
        :rtype: Tensor

        """
        x = self.cubic_layer1(feat_cubic)
        x = self.cubic_layer2(x)
        x = self.cubic_layer3(x)
        x = self.cubic_layer4(x)
        x = self.cubic_layer5(x)
        x = self.cubic_layer6(x)
        x = self.cubic_layer7(x)
        x = self.cubic_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)

        R = self.fc_cubic(x)

        # Normalised pixel coordinates in [0, 1): x_axis varies along dim 2 (H),
        # y_axis along dim 3 (W). Shared by all three filter branches.
        x_axis, x_axis2, x_axis3, y_axis, y_axis2, y_axis3 = _coord_grid_powers(
            img.shape[2], img.shape[3], img.device)

        # Evaluate Eq. 6 for every image in the batch and every RGB channel.
        # R has shape (B, 60): 20 coefficients per channel, r[0..19] = A..T in
        # the paper's order (x^3, x^2 y, x^2 i, x^2, x y^2, x y i, x y, x i^2,
        # x i, x, y^3, y^2 i, y^2, y i^2, y i, y, i^3, i^2, i, 1).
        # The x/y coordinate grids are shared across the batch and channels;
        # the coefficients are reshaped to (B, 3, 1, 1) so one expression
        # evaluates all three channels (this used to loop over c, with the
        # same per-element arithmetic).
        R = R.view(-1, 3, 20)
        r = [R[:, :, k].reshape(-1, 3, 1, 1) for k in range(20)]
        img_c = img  # (B, 3, H, W)
        cubic_mask = r[0] * x_axis3 + r[1] * x_axis2 * y_axis + r[2] * (
            x_axis2) * img_c + r[3] * x_axis2 + r[4] * x_axis * y_axis2 + r[
            5] * x_axis * y_axis * img_c \
            + r[6] * x_axis * y_axis + r[7] * x_axis * (img_c ** 2) + r[
            8] * x_axis * img_c + r[9] * x_axis + r[10] * (
            y_axis3) + r[11] * y_axis2 * img_c \
            + r[12] * y_axis2 + r[13] * y_axis * (img_c ** 2) + r[
            14] * y_axis * img_c + r[15] * y_axis + r[16] * (
            img_c ** 3) + r[17] * (img_c ** 2) \
            + r[18] * \
            img_c + r[19]

        img_cubic = torch.clamp(img + cubic_mask, 0, 1)
        return img_cubic


