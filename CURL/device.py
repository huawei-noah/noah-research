# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""Device selection: CUDA if present, then Apple Silicon (MPS), then CPU.

The original code called ``.cuda()`` directly, so it ran only on an NVIDIA GPU.
Everything now goes through ``DEVICE``.
"""
import torch


def _pick():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = _pick()


def empty_cache():
    """Free cached GPU memory, where the backend has a cache to free."""
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
