# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""Record the curves the network predicts, as training proceeds.

The curve layer predicts knot vectors, and what those knots look like over a run
is the question the layer exists to answer. Nothing in the training loop records
them, so this does: run a fixed image through the network, capture the output of
each curve head with a forward hook, and append a row to a JSON Lines file.

One fixed image throughout, so successive rows differ because the network
changed and not because the input did.
"""
import json
import os

import numpy as np
import torch

# The three heads and how many curves each predicts. fc_lab and fc_rgb emit
# three 16-knot curves; fc_hsv emits four.
HEADS = (('fc_lab', 3), ('fc_rgb', 3), ('fc_hsv', 4))
KNOTS = 16


class CurveLogger:
    """Captures predicted knots from a CURLNet and appends them to a file."""

    def __init__(self, net, log_dirpath, filename='curves.jsonl'):
        self.path = os.path.join(log_dirpath, filename)
        self._captured = {}
        self._handles = []
        layer = getattr(net, 'curllayer', None)
        if layer is None:                      # torch.compile wraps the module
            layer = getattr(getattr(net, '_orig_mod', net), 'curllayer', None)
        self.enabled = layer is not None
        if not self.enabled:
            return
        for name, _ in HEADS:
            head = getattr(layer, name, None)
            if head is None:
                self.enabled = False
                return
            self._handles.append(head.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inputs, output):
            self._captured[name] = output.detach().float().cpu()
        return hook

    def log(self, net, image, epoch):
        """Run one image through the net and append this epoch's knots."""
        if not self.enabled:
            return
        was_training = net.training
        net.eval()
        with torch.no_grad():
            net(image)
        if was_training:
            net.train()

        row = {'epoch': int(epoch)}
        for name, count in HEADS:
            out = self._captured.get(name)
            if out is None:
                continue
            # The knots are exp() of the head's output, which is what
            # apply_curve receives.
            knots = torch.exp(out[0, :count * KNOTS]).reshape(count, KNOTS)
            row[name] = [[round(float(v), 6) for v in c] for c in knots]
            # Curvature: zero for a straight line, which is what the smoothness
            # penalty drives the knots towards when nothing else trains them.
            second = np.diff(knots.numpy(), n=2, axis=1)
            row[name + '_curvature'] = round(float(np.abs(second).max()), 8)
            row[name + '_spread'] = round(
                float((knots.max() - knots.min()).item()), 6)
        with open(self.path, 'a') as f:
            f.write(json.dumps(row) + '\n')

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []
