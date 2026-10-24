# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""One DeepLPF training step, eager or captured in a CUDA graph per image shape.

At batch size 1 on a full-resolution FiveK image the step issues ~1,500
kernels, each a few microseconds of GPU work behind ~50 microseconds of
Python and dispatcher time, so the GPU idles for most of the step. A CUDA
graph replays the whole forward + loss + backward + Adam sequence from one
launch. FiveK has 16 distinct image sizes, so :class:`GraphedStep` keeps one
graph per input shape and captures each the first time the shape appears.

What the capture changes and what it does not:

* The arithmetic of every kernel is the eager arithmetic; prediction, loss
  and gradients are those of the eager step.
* Adam must run with ``capturable=True`` so its step count lives on the
  device. That path computes the bias corrections in fp32 tensor arithmetic
  rather than Python doubles, a ~1e-7 relative change in the effective step
  size.
* Each shape's warm-up runs a few real optimiser steps before capture (the
  allocator and cuDNN need them). Parameters and optimiser state are
  snapshotted before and restored after, so the trajectory is not advanced,
  but the dropout RNG stream is, so the dropout masks of a graphed run are
  not the masks of an eager run with the same seed. Same distribution,
  different draws.
"""
import torch



class EagerStep:
    """The plain training step: forward, loss, backward, Adam."""

    def __init__(self, net, criterion, optimizer, autocast_dtype=None,
                 gate_weight=None):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.autocast_dtype = autocast_dtype
        self.gate_weight = gate_weight

    def _autocast(self, device_type):
        if self.autocast_dtype is None:
            return torch.autocast(device_type, enabled=False)
        # No autocast cache: it keeps weight casts alive across steps and is
        # documented as unsafe under graph capture.
        return torch.autocast(device_type, dtype=self.autocast_dtype, cache_enabled=False)

    def forward_loss(self, x, y):
        # Autocast covers the network only. The loss stays fp32: its SSIM
        # variances are E[x^2] - mu^2, which cancels catastrophically in
        # bf16, and measured that way the parameter gradients moved by up to
        # 30% of their scale against fp32.
        with self._autocast(x.device.type):
            pred = self.net(x)
        loss = self.criterion(torch.clamp(pred.float(), 0.0, 1.0), y)
        # The L1 penalty on the per-instance gates, which is what makes the
        # number of active filters learned rather than fixed. Added here so it
        # is inside any CUDA-graph capture region, alongside the loss it joins.
        if self.gate_weight is not None:
            loss = loss + self.gate_weight * self.net.gate_penalty.float()
        return loss

    def __call__(self, x, y):
        """Run one step; returns the loss as a 0-d device tensor."""
        loss = self.forward_loss(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()


class GraphedStep(EagerStep):
    """:class:`EagerStep` captured in one CUDA graph per input shape.

    ``x`` and ``y`` are copied into per-shape static buffers and the shape's
    graph is replayed. The first occurrence of a shape costs ``warmup`` eager
    steps plus the capture; there are 16 shapes in FiveK.
    """

    def __init__(self, net, criterion, optimizer, autocast_dtype=None,
                 gate_weight=None, warmup=3):
        super().__init__(net, criterion, optimizer, autocast_dtype, gate_weight)
        self.warmup = warmup
        self.graphs = {}  # shape -> (static_x, static_y, static_loss, replay)

    # Split out so the bookkeeping can be tested without a GPU: a test
    # substitutes a capture that returns the step itself.
    def _capture(self, static_x, static_y):
        """Capture one step on the static buffers; return ``(static_loss, replay)``."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                EagerStep.__call__(self, static_x, static_y)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        # Grads must be allocated inside the capture so backward writes into
        # the graph's private pool on every replay (the pattern of the
        # torch.cuda.graphs whole-network example).
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(graph):
            loss = self.forward_loss(static_x, static_y)
            loss.backward()
            self.optimizer.step()
        return loss.detach(), graph.replay

    def _snapshot(self):
        params = [p.detach().clone() for p in self.net.parameters()]
        state = {p: {k: v.clone() for k, v in st.items() if torch.is_tensor(v)}
                 for p, st in self.optimizer.state.items()}
        return params, state

    def _restore(self, snapshot):
        params, state = snapshot
        with torch.no_grad():
            for p, saved in zip(self.net.parameters(), params):
                p.copy_(saved)
            for p, st in self.optimizer.state.items():
                saved = state.get(p, {})
                for k, v in st.items():
                    if not torch.is_tensor(v):
                        continue
                    # Adam's state is created on its first step, as zeros
                    # (exp_avg, exp_avg_sq, step): a param the snapshot did
                    # not know is back at its initial state when zeroed.
                    if k in saved:
                        v.copy_(saved[k])
                    else:
                        v.zero_()

    def __call__(self, x, y):
        key = (tuple(x.shape), tuple(y.shape))
        entry = self.graphs.get(key)
        if entry is None:
            static_x, static_y = x.clone(), y.clone()
            snapshot = self._snapshot()
            static_loss, replay = self._capture(static_x, static_y)
            self._restore(snapshot)
            entry = self.graphs[key] = (static_x, static_y, static_loss, replay)
        static_x, static_y, static_loss, replay = entry
        static_x.copy_(x, non_blocking=True)
        static_y.copy_(y, non_blocking=True)
        replay()
        return static_loss
