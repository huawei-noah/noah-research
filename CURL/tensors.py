# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""Small tensor-shape helpers shared by the colour and curve code.

Both the colour conversions and the curve application work on a channels-last
view and must accept either a single CHW image or a BCHW batch, so the rank
dispatch lives here rather than being repeated in each function.
"""
import torch


def channels_last(img: torch.Tensor) -> torch.Tensor:
    """Move the channel axis last, for a CHW image or a BCHW batch.

    The permutation is its own inverse, so the same call restores the original
    layout on the way out.

    :param img: CHW or BCHW tensor
    :returns: the same tensor with channels last
    """
    return img.permute(2, 1, 0) if img.dim() == 3 else img.permute(0, 3, 2, 1)


def monotone_mapping(params: torch.Tensor) -> torch.Tensor:
    """Build a strictly increasing tone mapping from unconstrained outputs.

    The default parameterisation reads the head's outputs as knot values of a
    scaling curve, which leaves nothing to stop the realised mapping
    ``x * scale(x)`` from decreasing: a brighter input pixel can map darker.

    Here the outputs are read as *increments* instead. softplus sends every one
    of them to a positive number, the cumulative sum is therefore strictly
    increasing, and normalising by the final value pins the curve to [0, 1].
    The mapping cannot invert, whatever the network emits, while the implied
    scaling factor stays free to rise or fall - so an edit that lifts shadows
    and rolls off highlights is still expressible.

    softplus rather than relu because relu is flat for negative inputs: a unit
    pushed negative would receive no gradient and never recover.

    :param params: (K,) or (B, K) of unconstrained head outputs
    :returns: the mapping's value at each of the K knots, increasing, ending at 1
    """
    steps = torch.nn.functional.softplus(params)
    curve = torch.cumsum(steps, dim=-1)
    return curve / curve[..., -1:].clamp(min=1e-8)


def split_curves(params: torch.Tensor, count: int) -> list:
    """Split a flat curve-parameter vector into ``count`` knot vectors.

    The heads emit their curves concatenated into one vector. The leading
    dimension is preserved, so a batch carries one curve per image.

    :param params: (N,) for a single image, or (B, N) for a batch
    :param count: how many curves the vector holds
    :returns: list of ``count`` knot tensors, exponentiated
    """
    width = params.shape[-1] // count
    return [torch.exp(params[..., i * width:(i + 1) * width])
            for i in range(count)]
