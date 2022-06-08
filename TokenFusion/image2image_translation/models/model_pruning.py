#Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.
#
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.runner import load_checkpoint
import math
from .modules import *
import cfg


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ModuleParallel(nn.Linear(in_features, hidden_features))
        self.dwconv = DWConv(hidden_features)
        self.act = ModuleParallel(nn.GELU())
        self.fc2 = ModuleParallel(nn.Linear(hidden_features, out_features))
        self.drop = ModuleParallel(nn.Dropout(drop))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = ModuleParallel(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = ModuleParallel(nn.Linear(dim, dim * 2, bias=qkv_bias))
        self.attn_drop = ModuleParallel(nn.Dropout(attn_drop))
        self.proj = ModuleParallel(nn.Linear(dim, dim))
        self.proj_drop = ModuleParallel(nn.Dropout(proj_drop))

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = ModuleParallel(nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio))
            self.norm = LayerNormParallel(dim)
        self.exchange = TokenExchange()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def softmax_with_mask(self, attn, mask, eps=1e-6):
        B, N, _ = mask.shape
        B, H, N, N = attn.shape
        attn_mask = mask.reshape(B, 1, 1, N)  # * mask.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_mask.dtype, device=attn_mask.device).view(1, 1, N, N)
        attn_mask = attn_mask + (1.0 - attn_mask) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_mask
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_mask.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, H, W, mask):
        # x: [B, N, C], mask: [B, N]
        B, N, C = x[0].shape
        q = self.q(x)
        q = [q_.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) for q_ in q]

        if self.sr_ratio > 1:
            tmp = [x_.permute(0, 2, 1).reshape(B, C, H, W) for x_ in x]
            tmp = self.sr(tmp)
            tmp = [tmp_.reshape(B, C, -1).permute(0, 2, 1) for tmp_ in tmp]
            kv = self.kv(self.norm(tmp))
        else:
            kv = self.kv(x)
        kv = [kv_.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) for kv_ in kv]
        k, v = [kv_[0] for kv_ in kv], [kv_[1] for kv_ in kv]

        attn = [(q_ @ k_.transpose(-2, -1)) * self.scale for (q_, k_) in zip(q, k)]
        attn = [attn_.softmax(dim=-1) for attn_ in attn]
        # if mask is None:
        #     attn = [attn_.softmax(dim=-1) for attn_ in attn]
        # else:
        #     attn = [self.softmax_with_mask(attn_, mask) for attn_ in attn]
        attn = self.attn_drop(attn)

        x = [(attn_ @ v_).transpose(1, 2).reshape(B, N, C) for (attn_, v_) in zip(attn, v)]
        x = self.proj(x)
        x = self.proj_drop(x)

        if mask is not None:
            x = [x_ * mask_.unsqueeze(2) for (x_, mask_) in zip(x, mask)]
            x = self.exchange(x, mask, mask_threshold=0.02)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = ModuleParallel(nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim))

    def forward(self, x, H, W):
        B, N, C = x[0].shape
        x = [x_.transpose(1, 2).view(B, C, H, W) for x_ in x]
        x = self.dwconv(x)
        x = [x_.flatten(2).transpose(1, 2) for x_ in x]

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., sr_ratio=1):
        super().__init__()
        self.norm1 = LayerNormParallel(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = ModuleParallel(DropPath(drop_path)) if drop_path > 0. else ModuleParallel(nn.Identity())
        self.norm2 = LayerNormParallel(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, mask=None):
        out = self.drop_path(self.attn(self.norm1(x), H, W, mask=mask))
        x = [x_ + out_ for (x_, out_) in zip(x, out)]
        out = self.drop_path(self.mlp(self.norm2(x), H, W))
        x = [x_ + out_ for (x_, out_) in zip(x, out)]
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = ModuleParallel(nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                   padding=(patch_size[0] // 2, patch_size[1] // 2)))
        self.norm = LayerNormParallel(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x[0].shape
        x = [x_.flatten(2).transpose(1, 2) for x_ in x]
        x = self.norm(x)

        return x, H, W


class PatchUpsample(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()

        self.proj = ModuleParallel(nn.Conv2d(in_chans // 4, embed_dim, kernel_size=1))
        self.norm = LayerNormParallel(embed_dim)
        self.pixelshuffle = ModuleParallel(nn.PixelShuffle(2))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.pixelshuffle(x)
        x = self.proj(x)
        B, C, H, W = x[0].shape
        x = [x_.flatten(2).transpose(1, 2) for x_ in x]

        return x, H, W


class PredictorLG(nn.Module):
    """ Image to Patch Embedding from DydamicVit
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        ) for _ in range(cfg.num_parallel)])

    def forward(self, x):
        x = [self.score_nets[i](x[i]) for i in range(cfg.num_parallel)]
        return x


class MixVisionTransformerGen(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., depths=[3, 4, 6, 3], sr_ratios=[16, 8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # transformer downsampling
        # patch_embed
        self.patch_embed_enc = nn.ModuleList()
        for i in range(5):
            patch_embed = OverlapPatchEmbed(img_size=img_size // 2 ** i, patch_size=3, stride=2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            self.patch_embed_enc.append(patch_embed)

        predictor_list = [PredictorLG(embed_dims[i]) for i in range(len(depths))]
        self.score_predictor = nn.ModuleList(predictor_list)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur_enc = 0
        self.block_enc, self.norm_enc = nn.ModuleList(), nn.ModuleList()
        for idx in [0, 1, 2, 3, 4]:
            block_enc = nn.ModuleList([Block(
                dim=embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_enc + i],
                sr_ratio=sr_ratios[idx])
                for i in range(depths[idx])])
            self.block_enc.append(block_enc)
            self.norm_enc.append(LayerNormParallel(embed_dims[idx]))
            cur_enc += depths[idx]

        # transformer upsampling
        # patch_embed
        self.patch_embed_dec = nn.ModuleList()
        for i in range(5)[::-1]:
            patch_embed = PatchUpsample(in_chans=embed_dims[i], embed_dim=embed_dims[max(0, i - 1)])
            self.patch_embed_dec.append(patch_embed)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur_dec = 0
        self.block_dec, self.norm_dec = nn.ModuleList(), nn.ModuleList()
        for idx in [3, 2, 1, 0]:
            block_dec = nn.ModuleList([Block(
                dim=embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_dec + i],
                sr_ratio=sr_ratios[idx])
                for i in range(depths[idx])])
            self.block_dec.append(block_dec)
            self.norm_dec.append(LayerNormParallel(embed_dims[idx]))
            cur_dec += depths[idx]

        self.project = ModuleParallel(nn.Conv2d(embed_dims[idx], 3, 1, 1, 0))
        self.tanh = ModuleParallel(nn.Tanh())
        self.alpha = nn.Parameter(torch.ones(cfg.num_parallel, requires_grad=True))
        self.register_parameter('alpha', self.alpha)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=None)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    def forward(self, x):
        B = x[0].shape[0]
        outs = []
        # x: torch.Size([1, 3, 256, 256])

        # Downsampling
        count, masks = 0, []
        for i in range(len(self.block_enc)):
            x, H, W = self.patch_embed_enc[i](x)
            # print('embed_enc %d:' % i, x[0].shape)
            for idx, blk in enumerate(self.block_enc[i]):
                mask = None
                if idx == len(self.block_enc[i]) - 1:
                    score = self.score_predictor[count](x)
                    # mask = [F.gumbel_softmax(score_.reshape(B, -1, 2), hard=True)[:, :, 0] for score_ in score]
                    mask = [F.softmax(score_.reshape(B, -1, 2), dim=2)[:, :, 0] for score_ in score]  # mask_: [B, N]
                    # print(count, mask[0].min(), mask[0].max(), mask[1].min(), mask[1].max(), mask[0].shape)
                    masks.append([mask_.flatten() for mask_ in mask])
                    count += 1
                x = blk(x, H, W, mask)
            # print('block_enc %d:' % i, x[0].shape)
            x = self.norm_enc[i](x)
            outs.append(x)
            x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]

        # Upsampling
        for i in range(len(self.block_dec)):
            x, H, W = self.patch_embed_dec[i](x)
            # print('embed_dec %d:' % i, x[0].shape)
            x = [x_ + outs_ for (x_, outs_) in zip(x, outs[::-1][i + 1])]
            for blk in self.block_dec[i]:
                x = blk(x, H, W)
            # print('block_dec %d:' % i, x[0].shape)
            x = self.norm_dec[i](x)
            x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]

        x, H, W = self.patch_embed_dec[4](x)
        # print('embed_enc 4:', x[0].shape)
        x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]

        # print(x[0].shape)
        x = self.tanh(self.project(x))
        ens = 0
        alpha_soft = F.softmax(self.alpha, dim=0)
        for l in range(cfg.num_parallel):
            ens += alpha_soft[l] * x[l].detach()
        x.append(ens)
        # print(x[0].shape)
        return x, alpha_soft, masks


class MixVisionTransformerDis(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=6, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., depths=[3, 4, 6, 3], sr_ratios=[16, 8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # transformer downsampling
        # patch_embed
        self.patch_embed_enc = nn.ModuleList()
        for i in range(5):
            patch_embed = OverlapPatchEmbed(img_size=img_size // 2 ** i, patch_size=3, stride=2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            self.patch_embed_enc.append(patch_embed)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur_enc = 0
        self.block_enc, self.norm_enc = nn.ModuleList(), nn.ModuleList()
        for idx in [0, 1, 2, 3, 4]:
            block_enc = nn.ModuleList([Block(
                dim=embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_enc + i],
                sr_ratio=sr_ratios[idx])
                for i in range(depths[idx])])
            self.block_enc.append(block_enc)
            self.norm_enc.append(LayerNormParallel(embed_dims[idx]))
            cur_enc += depths[idx]

        self.project = ModuleParallel(nn.Conv2d(embed_dims[idx], 1, 1, 1, 0))
        self.sigmoid = ModuleParallel(nn.Sigmoid())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=None)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    def forward(self, x, label):
        # x = [torch.cat([x_, label_], 1) for (x_, label_) in zip(x, label)]
        # print(x[0].shape)
        x = [torch.cat([x_, label_], 1) for (x_, label_) in zip(x, label)]
        B = x[0].shape[0]
        # x: torch.Size([1, 3, 256, 256])

        # Downsampling
        for i in range(len(self.block_enc)):
            x, H, W = self.patch_embed_enc[i](x)
            # print('embed_enc %d:' % i, x[0].shape)
            for blk in self.block_enc[i]:
                x = blk(x, H, W)
            # print('block_enc %d:' % i, x[0].shape)
            x = self.norm_enc[i](x)
            x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]

        # print(x[0].shape)
        x = self.project(x)
        x = self.sigmoid(x)
        # print('dis:', x[0].shape)
        return x
