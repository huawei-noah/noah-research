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



import logging
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from ops.modules import MSDeformAttn
from timm.models.layers import DropPath
from torch.nn.init import normal_


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points



class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., single=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features, single)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768, single=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.single = single

    def forward(self, x, H, W):
        if not self.single:
            B, N, C = x.shape
            n = N // 21
            x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
            x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
            x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
            x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
            x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
            x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
            x = torch.cat([x1, x2, x3], dim=1)
        else:
            B, N, C = x.shape
            x = x.transpose(1, 2).view(B, C, H, W).contiguous()
            x = self.dwconv(x).flatten(2).transpose(1, 2)
        return x


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False, single=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop, single=single)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        
        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn
    
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
            
        return query


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        
        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            return query + self.gamma * attn
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
            
        return query


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        
        def _inner_forward(x):
            c1 = self.stem(x) # b 64 256 256
            c2 = self.conv2(c1) # b 128 128 128
            c3 = self.conv3(c2) # b 256 64 64
            c4 = self.conv4(c3) # b 256 32 32
            c1 = self.fc1(c1) # b 128 256 256
            c2 = self.fc2(c2) # b 128 128 128
            c3 = self.fc3(c3) # b 128 64 64
            # c4 = self.fc4(c4) # b 128 32 32
    
            bs, dim, _, _ = c1.shape
            c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 2s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 4s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 8s
            # c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 16s
    
            return c1, c2, c3
        
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs


class InteractionBlockPrompt(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False, next_dim=64, stride=2):
        super().__init__()

        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)

        self.next_dim = next_dim
        if self.next_dim is not None:
            self.up_layer1 = nn.Conv2d(dim, self.next_dim, kernel_size=3, stride=stride, padding=1, bias=False)
            self.up_layer2 = nn.Conv2d(dim, self.next_dim, kernel_size=3, stride=stride, padding=1, bias=False)
            self.up_layer3 = nn.Conv2d(dim, self.next_dim, kernel_size=3, stride=stride, padding=1, bias=False)


    def deform_inputs(self, hw_shape, device):
        h, w = hw_shape
        h, w = h*4, w*4
        spatial_shapes = torch.as_tensor([(h // 2, w // 2),
                                        (h // 4, w // 4),
                                        (h // 8, w // 8)],
                                        dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = get_reference_points([(h // 4, w //4)], device)

        deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
        
        spatial_shapes = torch.as_tensor([(h // 4, w // 4)], dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = get_reference_points([(h // 2, w // 2),
                                                (h // 4, w // 4),
                                                (h // 8, w // 8)], device)
        deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
        return deform_inputs1, deform_inputs2

    def forward(self, x, c1, c2, c3, stage, hw_shape, backbone='swin', vgg_layers=None):

        H, W = hw_shape
        bs, channel, dim = x.shape
        deform_inputs1, deform_inputs2 = self.deform_inputs(hw_shape, x.device) 

        c = torch.cat([c1, c2, c3], dim=1) # b*86016*dim

        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)

        x_injector = self.injector(query=x, reference_points=deform_inputs1[0],
                                feat=c, spatial_shapes=deform_inputs1[1],
                                level_start_index=deform_inputs1[2])
       
        c1 = c1.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c2 = c2.transpose(1, 2).view(bs, dim, H, W ).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()

        if self.next_dim is not None:
            c1 = self.up_layer1(c1)
            c2 = self.up_layer2(c2)
            c3 = self.up_layer3(c3)

            c1 = c1.view(bs, self.next_dim, -1).transpose(1, 2)  # 2s
            c2 = c2.view(bs, self.next_dim, -1).transpose(1, 2)  # 4s
            c3 = c3.view(bs, self.next_dim, -1).transpose(1, 2)  # 8s
       
        if backbone == 'swin':
            x_stage, hw_shape_stage, out_stage, out_hw_shape_stage = stage(x_injector, hw_shape)
            x_injector = x_injector.view(bs, channel, dim).transpose(1, 2).view(bs, dim, H, W)
            return x_injector, c1, c2, c3, x_stage, hw_shape_stage, out_stage, out_hw_shape_stage
        
        elif backbone == 'mit':
            prompt = x_injector
            for i, blk in enumerate(stage):
                x_injector = blk(x_injector, H, W)
            prompt = prompt.view(bs, channel, dim).transpose(1, 2).view(bs, dim, H, W)
            return prompt, c1, c2, c3, x_injector

        else:
            raise EOFError()





