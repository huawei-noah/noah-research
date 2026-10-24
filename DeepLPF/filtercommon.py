# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""Pieces the three filter heads share: coordinate grids, the STE, the gates.

The grids are cached because every head builds the same ones for a given image
size. ``BinaryLayer`` is the straight-through binarisation of Sec. 3.2.2, used
by the graduated filter, and ``_apply_gates`` scales one filter instance's
deviation from neutral by its predicted gate (the optional
``--learn_filter_count`` feature).
"""
import torch
import torch.nn as nn


#: Instances per filter branch, and so the number of gates when the filter
#: count is learned.
GATES_PER_BRANCH = 3

#: Coordinate grids keyed by (H, W, device); every filter head builds the same
#: ones for a given image size, so they are built once and reused.
_GRID_CACHE = {}


def _coord_grids(H, W, device):
    """Normalised pixel-coordinate grids shared by the three filter branches.

    ``x_axis`` varies along dim 2 (H), ``y_axis`` along dim 3 (W), both in
    [0, 1). Built exactly as the branches used to build them inline, but
    cached per (H, W, device): FiveK has 16 distinct image sizes, and building
    them per call cost six host-to-device copies per forward pass.
    """
    key = (H, W, device)
    grids = _GRID_CACHE.get(key)
    if grids is None:
        x_axis = torch.arange(H).view(-1, 1).repeat(1, W).to(device) / H
        y_axis = torch.arange(W).repeat(H, 1).to(device) / W
        grids = _GRID_CACHE[key] = (x_axis, y_axis)
    return grids


def _coord_grid_powers(H, W, device):
    """``(x, x^2, x^3, y, y^2, y^3)`` of the grids from :func:`_coord_grids`.

    The cubic filter's polynomial needs the squares and cubes of the
    coordinate grids; they depend only on the image size, so compute them
    once per size rather than on every forward pass.
    """
    key = ('pow', H, W, device)
    powers = _GRID_CACHE.get(key)
    if powers is None:
        x_axis, y_axis = _coord_grids(H, W, device)
        powers = _GRID_CACHE[key] = (
            x_axis, x_axis ** 2, x_axis ** 3, y_axis, y_axis ** 2, y_axis ** 3)
    return powers


class SignSTE(torch.autograd.Function):
    """``sign`` with a straight-through gradient.

    ``torch.sign`` has zero gradient everywhere, so a binarisation layer built
    on it alone blocks learning. This is the estimator of Courbariaux et al.:
    the forward pass binarises, and the backward pass passes the gradient
    through unchanged except where the input has saturated (``|input| > 1``),
    where it is zeroed.

    Implemented as an ``autograd.Function`` because that is the only thing
    PyTorch calls ``backward`` on -- see :class:`BinaryLayer`.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (input.abs() <= 1).to(grad_output.dtype)


class BinaryLayer(nn.Module):
    """Binarisation layer with a straight-through estimator.

    Used by the graduated filter for the binarised inversion indicator
    ``g_inv_hat = (sgn(g_inv) + 1) / 2`` of paper Sec. 3.2.2.

    The estimator lives in :class:`SignSTE`, a ``torch.autograd.Function``:
    ``torch.sign`` has zero gradient everywhere, so the binarisation has to pass
    the gradient through itself for the inversion indicators to train at all.
    """

    def forward(self, input):
        """Binarise the input with a straight-through gradient.

        :param input: data
        :returns: sign of data
        :rtype: Tensor

        """
        return SignSTE.apply(input)


def _apply_gates(mask_scale, gates):
    """Scale each instance's deviation from neutral by its gate.

    ``1 + g * (s - 1)`` is the identity at ``g = 0`` and leaves ``s`` untouched
    at ``g = 1``. Since instances fuse by multiplication and 1 is the
    multiplicative identity, a gate at zero removes its instance from the
    product exactly, at no cost to the remaining ones - which is what makes an
    L1 penalty on the gates a penalty on the number of active filters.

    :param mask_scale: per-instance scaling maps, (B, instance, channel, H, W)
    :param gates: gate per instance in [0, 1], (B, instance)
    :returns: gated scaling maps, same shape as ``mask_scale``
    :rtype: Tensor

    """
    g = gates.view(-1, gates.shape[1], 1, 1, 1)
    return 1.0 + g * (mask_scale - 1.0)
