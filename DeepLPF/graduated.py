# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""The graduated filter of Sec. 3.2.2, Eqs. 1-3.

It predicts pairs of parallel lines and scales the image between them, with the
inversion indicator binarised through the straight-through estimator.
"""
import torch
import torch.nn as nn

from blocks import ConvBlock, GlobalPoolingBlock, MaxPoolBlock
from filtercommon import (GATES_PER_BRANCH, BinaryLayer, _apply_gates,
                          _coord_grids)


class GraduatedFilter(nn.Module):
    """Graduated-filter branch of DeepLPF (paper Sec. 3.2.2, Eqs. 1-3; Sec. 3.3, Eq. 7).

    Emulates a photographic graduated filter: three parallel lines split the
    image into a 100% region, two linear-decay bands and a 0% region. The
    network regresses ``G = 24`` values: for each of three filter instances a
    binarised inversion indicator ``g_inv`` (via :class:`BinaryLayer`), the
    central-line slope ``m``, two offsets ``o1, o2`` (so ``d1 = o1 cos(alpha)``,
    ``d2 = o2 cos(alpha)``, ``alpha = atan(m)``), and three per-channel scaling
    factors ``s_g^R, s_g^G, s_g^B``. Each instance yields a (B, 3, H, W)
    scaling map; the three instances are fused by element-wise multiplication
    (Eq. 7) into ``s_g``. Scalings are bounded to [0, max_scale = 2].
    """

    def __init__(self, num_in_channels=64, num_out_channels=64,
                 learn_filter_count=False):
        """Initialisation function for the graduated filter

        :param num_in_channels:  input channels
        :param num_out_channels: output channels
        :returns: N/A
        :rtype: N/A

        """
        super(GraduatedFilter, self).__init__()
        self.learn_filter_count = learn_filter_count

        self.graduated_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.graduated_layer2 = MaxPoolBlock()
        self.graduated_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer4 = MaxPoolBlock()
        self.graduated_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer6 = MaxPoolBlock()
        self.graduated_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer8 = GlobalPoolingBlock()
        # 24 filter parameters, plus one gate per instance when the filter
        # count is learned. The extra outputs change the layer shape, so gated and ungated checkpoints are not interchangeable.
        self.fc_graduated = torch.nn.Linear(
            num_out_channels, 24 + (GATES_PER_BRANCH if learn_filter_count else 0))
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear',align_corners=False)
        self.dropout = nn.Dropout(0.5)
        self.bin_layer = BinaryLayer()

    def tanh01(self, x):
        """Adjust Tanh to return values between 0 and 1

        :param x: Tensor arbitrary range
        :returns: Tensor between 0 and 1
        :rtype: tensor

        """
        tanh = nn.Tanh()
        return 0.5 * (tanh(x) + 1)

    def get_inverted_mask(self, factor, invert, d1, d2, max_scale, top_line):
        """Builds one channel's graduated scaling map ``s(x, y)`` (paper Eqs. 1-3).

        ``top_line`` plays the role of the signed distance ``l(x, y)`` to the
        central line; the two ramps ``(... / d1) * top_line`` and
        ``(... / d2) * top_line`` are the linear decays of Eqs. 1 and 2, and
        ``invert`` is the binarised ``g_inv`` of Eq. 3 that swaps which side of
        the line receives the full scaling. The four cases (inverted or not,
        scale factor above or below 1) are the branches of Eq. 3 written out
        so that the scaling always ramps between 1 (no change) and ``factor``.

        All arguments broadcast against each other; the per-image scalars are
        passed with trailing singleton dims so they broadcast over the (H, W)
        grid. The caller evaluates all nine (instance, channel) masks in one
        call with ``factor`` of shape (B, 3, 3, 1, 1), the per-instance
        ``invert``, ``d1``, ``d2`` of shape (B, 3, 1, 1, 1) and ``top_line`` of
        shape (B, 3, 1, H, W); the arithmetic per element is exactly what a
        single (B, H, W) call performed.

        :param factor: scale factor s_g
        :param invert: binary indicator variable g_inv in {0, 1}
        :param d1: distance between top and mid line
        :param d2: distance between bottom and mid line
        :param max_scale: maximum scaling factor possible
        :param top_line: soft above/below-line map
        :returns: scaling mask, the broadcast shape of the inputs
        :rtype: Tensor

        """
        # The original code branched with `(invert == 1).all()` / `(factor >= 1).all()`,
        # which collapses the batch to a single decision. To support a batch whose
        # images take different branches, all four branch expressions are evaluated
        # and selected per image with torch.where; the clamp bounds (which depend
        # only on factor >= 1) are likewise applied element-wise.
        f = factor
        fac_ge1 = (factor >= 1)

        # Every branch has the form  base + (A / d1) * top_line + (B / d2) * top_line
        # with per-image scalars base, A, B; only those differ between the
        # branches. Select them per image first, then evaluate the ramp once
        # at full resolution instead of four times. The selected map is
        # arithmetically identical to the branch it came from.
        diff_hi = (f - 1) / 2 + 1
        diff_lo = (1 - f) / 2 + f
        # invert == 1
        A_inv = torch.where(fac_ge1, diff_hi - f, diff_lo - f)
        B_inv = torch.where(fac_ge1, 1 - diff_hi, 1 - diff_lo)
        # invert != 1
        A_non = torch.where(fac_ge1, diff_hi - f, diff_lo - 1)
        B_non = torch.where(fac_ge1, f - diff_hi, f - diff_lo)

        def ramp(base, A, B):
            return base + (A / d1) * top_line + (B / d2) * top_line

        branch_inv = ramp(f, A_inv, B_inv)
        branch_non = ramp(1, A_non, B_non)

        # A torch.where CONDITION is not differentiable, so selecting on
        # `invert == 1` severs g_inv from the loss - a second cut, one line
        # after the one the `ste` fix repairs. Even with a working
        # straight-through estimator the inversion indicators receive no
        # gradient.
        #
        # Blend the two branches with the binarised indicator instead. The
        # forward value is unchanged, because `invert` is exactly 0 or 1, but
        # now the STE's gradient reaches g_inv.
        w = invert.to(branch_inv.dtype)
        mask_scale = w * branch_inv + (1.0 - w) * branch_non

        # factor >= 1 branches clamp to [1, max_scale]; factor < 1 branches to [0, 1].
        lower = torch.where(fac_ge1, torch.ones_like(f), torch.zeros_like(f))
        upper = torch.where(fac_ge1, torch.full_like(f, float(max_scale)), torch.ones_like(f))
        mask_scale = torch.minimum(torch.maximum(mask_scale, lower), upper)

        mask_scale = torch.clamp(mask_scale, 0, max_scale)
        return mask_scale

    def get_graduated_mask(self, feat, img):
        """ Graduated filter definition

        :param feat: features
        :param img: image
        :returns: scaling map
        :rtype: Tensor

        """
        feat_graduated = torch.cat((feat, img), 1)
        feat_graduated = self.upsample(feat_graduated)
        return self.mask_from_input(feat_graduated, img)

    def mask_from_input(self, feat_graduated, img):
        """As :meth:`get_graduated_mask`, given the already resized 300x300 input.

        :param feat_graduated: (B, 64, 300, 300) resized concatenation of features and image
        :param img: image the filter is applied to, (B, 3, H, W)
        :returns: scaling map, (B, 3, H, W)
        :rtype: Tensor

        """

        # Normalised pixel coordinates (see CubicFilter.get_cubic_mask)
        x_axis, y_axis = _coord_grids(img.shape[2], img.shape[3], img.device)

        # Parameter prediction (Sec. 3.2.1): image + backbone features -> 24 values
        x = self.graduated_layer1(feat_graduated)
        x = self.graduated_layer2(x)
        x = self.graduated_layer3(x)
        x = self.graduated_layer4(x)
        x = self.graduated_layer5(x)
        x = self.graduated_layer6(x)
        x = self.graduated_layer7(x)
        x = self.graduated_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        G = self.fc_graduated(x)

        # G has shape (B, 24); every parameter below is a per-image vector of
        # shape (B,). Layout: [0:3] g_inv, [3:6] slope m, [6:9] central-line
        # intercept c, [9:12] offset o1, [12:15] offset o2 (as a fraction of
        # o1, so 0 <= o2 <= o1 holds by construction),
        # [15:24] per-channel scale factors s_g (three instances x R,G,B).
        #
        # The three instances are evaluated together along a leading "instance"
        # dim: every per-instance quantity below has shape (B, 3), and the nine
        # (instance, channel) masks come out of one get_inverted_mask call as
        # (B, 3, 3, H, W). This replaced three copies of every line and nine
        # mask calls; the per-element arithmetic is unchanged.
        #
        # Inversion indicator g_inv_hat = (sgn(g_inv) + 1) / 2 in {0, 1} (Sec. 3.2.2)
        above_or_below_line = ((self.bin_layer(G[:, 0:3]))+1)/2

        slope = G[:, 3:6].clone()

        # Central-line intercept c, and the two offsets o1 >= o2 >= 0.
        #
        # These three lines used to read:
        #
        #     y_axis_dist = tanh01(G[:, 6:9]) + eps
        #     y_axis_dist = clamp(maximum(tanh01(G[:, 9:12]), y_axis_dist.data), max=1)
        #     y_axis_dist_lo = clamp(minimum(tanh01(G[:, 12:15]), y_axis_dist.data), min=0)
        #
        # G[:, 6:9] was consumed only through `.data`, which strips the
        # autograd graph, so those three output units received exactly zero
        # gradient while still setting the forward value (the floor under o1).
        # In the shipped checkpoint they are still at their initialisation,
        # tanh01(~0) ~= 0.5, and that untrained constant wins the maximum on
        # two of the three filter instances for every bundled example: the top
        # offset of those instances was a frozen 0.5, not a prediction.
        #
        # Removing `.data` alone is not enough. maximum/minimum route gradient
        # to whichever argument is selected, so G[:, 6:9] and G[:, 9:12] would
        # take turns and each still be starved on any step where the other
        # wins (measured: rows 6-8 still all-zero in 2 of 8 seeds).
        #
        # A learned floor under an already-free value in (0, 1) also buys no
        # expressiveness - o1 can simply predict the larger number itself. So
        # the three outputs are given the job the line geometry was missing.
        # The intercept c was conflated with o1 (the central line sat at
        # y = m x + o1 + d1), which locked the line's position to the width of
        # its own ramp; Eqs. 1-3 have them separate. G[:, 6:9] is now c, and
        # o2 is predicted as a fraction of o1 so the ordering o2 <= o1 holds by
        # construction. Every output is now on an unconditional differentiable
        # path and both clamps are gone.
        intercept = self.tanh01(G[:, 6:9])
        y_axis_dist = self.tanh01(G[:, 9:12])
        y_axis_dist_lo = y_axis_dist * self.tanh01(G[:, 12:15])

        # Scales: (B, 9) laid out instance-major, i.e. [inst0 R,G,B, inst1 R,G,B, inst2 R,G,B]
        max_scale = 2

        scale_factor = self.tanh01(G[:, 15:24]) * max_scale

        slope_angle = torch.atan(slope)

        # Distances between the central line and the two outer lines:
        # d1 = o1 cos(alpha), d2 = o2 cos(alpha), alpha = atan(m) (Sec. 3.2.2)
        d_top = self.tanh01(y_axis_dist*torch.cos(slope_angle))
        d_bot = self.tanh01(y_axis_dist_lo*torch.cos(slope_angle))

        # Soft side-of-line map: tanh01 of l(x, y) = y - (m x + c) shifted by d1,
        # i.e. ~1 above the top line, ~0 below, ramping in between.
        # Broadcast the per-image line parameters (B, 3) over the (H, W) grid -> (B, 3, H, W)
        top_line = self.tanh01(y_axis - (slope.view(-1, 3, 1, 1) * x_axis + intercept.view(-1, 3, 1, 1) + d_top.view(-1, 3, 1, 1)))

        # Three filter instances, each with a per-channel (R, G, B) scaling map:
        # (B, instance, channel, H, W).
        mask_scale = self.get_inverted_mask(
            scale_factor.view(-1, 3, 3, 1, 1), above_or_below_line.view(-1, 3, 1, 1, 1),
            d_top.view(-1, 3, 1, 1, 1), d_bot.view(-1, 3, 1, 1, 1), max_scale,
            top_line.unsqueeze(2))
        mask_scale = torch.clamp(mask_scale, 0, max_scale)

        # `gates` feature: one learned gate per instance, so the branch can use
        # fewer than three filters. Applied before the product, where a gate at
        # zero makes its instance exactly 1 and drops out of the fuse.
        if self.learn_filter_count:
            self.gates = self.tanh01(G[:, 24:27])
            mask_scale = _apply_gates(mask_scale, self.gates)

        # Fuse the three instances by element-wise multiplication: s_g = prod_i s_gi (Eq. 7)
        mask_scale = torch.clamp(
            mask_scale[:, 0]*mask_scale[:, 1]*mask_scale[:, 2], 0, max_scale)

        return mask_scale


