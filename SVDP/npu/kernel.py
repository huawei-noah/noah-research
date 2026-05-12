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