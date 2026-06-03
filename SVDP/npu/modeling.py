# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch


def up_proj(w, x):
    return torch.mm(w, x)


def down_proj(w, x):
    return torch.mm(w, x)


def up_proj_sparse(w, x, idx):
    ws = torch.index_select(w, 0, idx)
    return torch.mm(ws, x)


def down_proj_sparse(w, x, idx):
    ws = torch.index_select(w, 0, idx)
    return torch.mm(ws.t(), x)


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        a, b = 0.1, 0.05
        self.gate_w = torch.rand((hidden_dim, dim)).to(torch.float16).npu() * a - b
        self.proj_up_w = torch.rand((hidden_dim, dim)).to(torch.float16).npu() * a - b
        self.proj_down_w = torch.rand((dim, hidden_dim)).to(torch.float16).npu() * a - b
        self.hidden_dim = hidden_dim

    def zero_weights(self, mask):
        for i in range(self.hidden_dim):
            if not mask[i]:
                self.gate_w[i, :] = 0
                self.proj_up_w[i, :] = 0

    @torch.inference_mode()
    def forward(self, x):
        g = torch.mm(self.gate_w, x)
        u = torch.mm(self.proj_up_w, x)
        gu = torch.nn.functional.relu(g) * u
        return torch.mm(self.proj_down_w, gu)


class SparseFeedForward(torch.nn.Module):
    def __init__(self, ff: FeedForward):
        super().__init__()
        self.gate_w = ff.gate_w.clone()
        self.proj_up_w = ff.proj_up_w.clone()
        self.proj_down_w = ff.proj_down_w.clone().t().contiguous()

    @torch.inference_mode()
    def forward(self, x, mask):
        g = up_proj_sparse(self.gate_w, x, mask)
        u = up_proj_sparse(self.proj_up_w, x, mask)
        gu = torch.nn.functional.relu(g) * u
        return down_proj_sparse(self.proj_down_w, gu, mask)
