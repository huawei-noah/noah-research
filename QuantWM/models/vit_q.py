# adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
from torch import nn
from einops import rearrange
from .ptq import QAct, QLinear, QConv2d,QLayerNorm

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
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask

class QFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.,
                 quant=False,
                 calibrate=False,
                 cfg=None):
        super().__init__()
        self.net = nn.Sequential(
            QLayerNorm(dim),  
            QLinear(dim, hidden_dim,
                    quant=quant,
                    calibrate=calibrate,
                    bit_type=cfg.BIT_TYPE_W,
                    calibration_mode=cfg.CALIBRATION_MODE_W,
                    observer_str=cfg.OBSERVER_W,
                    quantizer_str=cfg.QUANTIZER_W),
            nn.GELU(),          
            nn.Dropout(dropout),
            QLinear(hidden_dim, dim,
                    quant=quant,
                    calibrate=calibrate,
                    bit_type=cfg.BIT_TYPE_W,
                    calibration_mode=cfg.CALIBRATION_MODE_W,
                    observer_str=cfg.OBSERVER_W,
                    quantizer_str=cfg.QUANTIZER_W),
            nn.Dropout(dropout) 
        )

        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

    def forward(self, x):
        for i, module in enumerate(self.net):
            if i == 1:
                x = module(self.qact1(x))
            elif i == 4:
                x = module(self.qact2(x))
            else:
                x = module(x)
        return x

class QAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,
                 quant=False,
                 calibrate=False,
                 cfg=None,
                 *,
                 num_patches: int,
                 num_frames: int):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = QLayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = QLinear(dim, inner_dim * 3, bias=False,
                            quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W)

        self.to_out = nn.Sequential(
            QLinear(inner_dim, dim,
                    quant=quant,
                    calibrate=calibrate,
                    bit_type=cfg.BIT_TYPE_W,
                    calibration_mode=cfg.CALIBRATION_MODE_W,
                    observer_str=cfg.OBSERVER_W,
                    quantizer_str=cfg.QUANTIZER_W),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        bias = generate_mask_matrix(num_patches, num_frames)
        self.register_buffer("bias", bias, persistent=False)

        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

    def forward(self, x):
        B, T, C = x.size()
        x = self.norm(x)
        x = self.qact1(x)

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
        out = self.qact2(out)
        return self.to_out(out)

class QTransformerBlock(nn.Module):
    """替代 nn.ModuleList([QAttention, QFeedForward])"""
    def __init__(self, attn: QAttention, ff: QFeedForward):
        super().__init__()
        self.attn = attn
        self.ff = ff

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

class QTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,
                 quant=False,
                 calibrate=False,
                 cfg=None,
                 *,
                 num_patches: int,
                 num_frames: int):
        super().__init__()
        self.norm = QLayerNorm(dim)
        self.layers = nn.ModuleList([
            QTransformerBlock(
                QAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                           quant=quant, calibrate=calibrate, cfg=cfg,
                           num_patches=num_patches, num_frames=num_frames),
                QFeedForward(dim, mlp_dim, dropout=dropout,
                             quant=quant, calibrate=calibrate, cfg=cfg),
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return self.norm(x)

class QViTPredictor(nn.Module):
    def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim,
                 pool='cls', dim_head=64, dropout=0., emb_dropout=0.,
                 quant=False,
                 calibrate=False,
                 cfg=None):
        super().__init__()
        assert pool in {'cls', 'mean'}

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = QTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout,
            quant=quant, calibrate=calibrate, cfg=cfg,
            num_patches=num_patches, num_frames=num_frames
        )

        self.pool = pool
        self.cfg = cfg

    def model_quant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct]:
                m.quant = True

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct]:
                m.calibrate = False

    def forward(self, x):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x