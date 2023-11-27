# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Modified from: https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0

import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
from ops.modules import MSDeformAttn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_

from mmseg.models.backbones.swin import SwinTransformer
from .adapter_modules import SpatialPriorModule
from .adapter_modules import InteractionBlockPrompt


@BACKBONES.register_module()
class SwinPromptBase(SwinTransformer):
    def __init__(self,
                 freeze_backbone=True,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 conv_inplane=64,
                 n_points=4,
                 deform_num_heads=16,
                 init_values=1e-6,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 deform_ratio=0.5,
                 norm_layer=None,
                 use_extra_extractor=False,
                 pretrained=None,
                 frozen_stages=-1,
                 init_cfg=None,):
        super().__init__(
                 pretrain_img_size=pretrain_img_size, in_channels=in_channels, embed_dims=embed_dims, patch_size=patch_size,
                 window_size=window_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, strides=strides,
                 out_indices=out_indices, qkv_bias=qkv_bias, qk_scale=qk_scale, patch_norm=patch_norm, drop_rate=drop_rate,
                 attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, use_abs_pos_embed=use_abs_pos_embed,
                 act_cfg=act_cfg, norm_cfg=norm_cfg, with_cp=with_cp, pretrained=None, frozen_stages=frozen_stages, init_cfg=init_cfg)
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.next_dim = [256, 512, 1024, None]
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dims, with_cp=False)

        self.interactions = nn.Sequential(*[
            InteractionBlockPrompt(dim=embed_dims*2**i, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=drop_path_rate,
                             norm_layer=norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=use_extra_extractor,
                             with_cp=with_cp, next_dim=self.next_dim[i])
            for i in range(len(self.stages))
        ])

        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)

        if freeze_backbone:
            for name, params in self.named_parameters():
                if 'spm' in name or 'interactions' in name:
                    params.requires_grad=True
                else:
                    params.requires_grad=False

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x):

        # SPM forward
        c1, c2, c3 = self.spm(x) # b*65536*128 b*16384*128 b*4096*128 b*1024*128
        x, hw_shape = self.patch_embed(x) # b*21504*128
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        # Interaction with spm prompt
        outs = []

        for i, (layer, stage) in enumerate(zip(self.interactions, self.stages)):
        
            x_injector, c1, c2, c3, x, hw_shape, out, out_hw_shape = layer(x, c1, c2, c3, stage, hw_shape, backbone='swin')

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs


@BACKBONES.register_module()
class SwinPrompt(SwinPromptBase):
    def __init__(self,
                 freeze_backbone=True,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 conv_inplane=64,
                 n_points=4,
                 deform_num_heads=16,
                 init_values=1e-6,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 deform_ratio=0.5,
                 norm_layer=None,
                 use_extra_extractor=False,
                 pretrained=None,
                 frozen_stages=-1,
                 init_cfg=None,):
        super().__init__(
                freeze_backbone=freeze_backbone,pretrain_img_size=pretrain_img_size,in_channels=in_channels,embed_dims=embed_dims,patch_size=patch_size,window_size=window_size,
                 mlp_ratio=mlp_ratio,depths=depths,num_heads=num_heads,strides=strides,out_indices=out_indices,qkv_bias=qkv_bias,qk_scale=qk_scale,
                 patch_norm=patch_norm,drop_rate=drop_rate,attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,use_abs_pos_embed=use_abs_pos_embed,
                 act_cfg=act_cfg,norm_cfg=norm_cfg,with_cp=with_cp,conv_inplane=conv_inplane,n_points=n_points,deform_num_heads=deform_num_heads,init_values=init_values,
                 with_cffn=with_cffn,cffn_ratio=cffn_ratio,deform_ratio=deform_ratio,norm_layer=norm_layer,use_extra_extractor=use_extra_extractor,pretrained=pretrained,
                 frozen_stages=frozen_stages,init_cfg=init_cfg)
    
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dims))
        normal_(self.level_embed)

        if freeze_backbone:
            for name, params in self.named_parameters():
                if 'spm' in name or 'interactions' in name or 'level_embed' in name:
                    params.requires_grad=True
                else:
                    params.requires_grad=False

    def _add_level_embed(self, c1, c2, c3):
        c1 = c1 + self.level_embed[0]
        c2 = c2 + self.level_embed[1]
        c3 = c3 + self.level_embed[2]
        return c1, c2, c3

    def forward(self, x):
        # SPM forward
        out_prompt = []
        c1, c2, c3 = self.spm(x) # b*65536*128 b*16384*128 b*4096*128 b*1024*128
        c1, c2, c3 = self._add_level_embed(c1, c2, c3)
        # out_prompt.append(torch.cat((c1, c2, c3), dim=1))
        x, hw_shape = self.patch_embed(x) # b*21504*128
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        # Interaction with spm prompt
        outs = []
        for i, (layer, stage) in enumerate(zip(self.interactions, self.stages)):
            x_injector, c1, c2, c3, x, hw_shape, out, out_hw_shape = layer(x, c1, c2, c3, stage, hw_shape, backbone='swin')
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
                # if i != len(self.stages) - 1:
                out_prompt.append(x_injector)
        outputs = dict()
        outputs["outs"] = outs
        outputs["prompts"] = out_prompt
        return outputs



