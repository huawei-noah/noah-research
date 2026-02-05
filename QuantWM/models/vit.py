# adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
from torch import nn
from einops import rearrange

# helpers
NUM_FRAMES = 1
NUM_PATCHES = 1

def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
    return mask

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),             # 0
            nn.Linear(dim, hidden_dim),    # 1
            nn.GELU(),                     # 2
            nn.Dropout(dropout),           # 3
            nn.Linear(hidden_dim, dim),    # 4
            nn.Dropout(dropout),           # 5
        )

    def forward(self, x):
        for module in self.net:
            x = module(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., *, num_patches: int, num_frames: int):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        bias = generate_mask_matrix(num_patches, num_frames)  # float tensor
        self.register_buffer("bias", bias, persistent=False)

    def forward(self, x):
        B, T, C = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        bias = self.bias[:, :, :T, :T]
        if bias.device != dots.device:
            bias = bias.to(dots.device)
        dots = dots.masked_fill(bias == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., *, num_patches: int, num_frames: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                          num_patches=num_patches, num_frames=num_frames),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))


    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViTPredictor(nn.Module):
    def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim,
                 pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout,
            num_patches=num_patches, num_frames=num_frames
        )

        self.pool = pool

    def forward(self, x):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x
